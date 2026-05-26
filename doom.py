#!/usr/bin/env python3
"""
Doom -> two Framework Laptop 16 LED Matrix displays as one optimized 18x34 grayscale display.

Two 9x34 modules are stitched side-by-side into one 18x34 canvas.

This version keeps the original safe serial behavior:
- One 0x07 stage-column packet per serial write.
- One 0x08 flush packet after changed columns.
- Optional parallel writes across multiple matrix modules with --serial-workers.
- Optional experimental packet bursting with --serial-burst, disabled by default.

Dependencies:
    python -m pip install --user --break-system-packages pyserial pygame pillow numpy vizdoom

Run:
    python doom.py --iwad DOOM.WAD

Recommended first test:
    python doom.py --iwad DOOM.WAD --fps 14 --serial-workers 1 --preview-scale 7

Then try:
    python doom.py --iwad DOOM.WAD --fps 24 --serial-workers 2 --preview-scale 7

For more FPS after confirming LEDs work:
    python doom.py --iwad DOOM.WAD --fps 30 --serial-workers 2 --preview-scale 0 --no-autocontrast --levels 16 --min-change 4 --dither none
"""

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import serial
import serial.tools.list_ports
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


FWK_MAGIC = bytes([0x32, 0xAC])

CMD_BRIGHTNESS = 0x00
CMD_SLEEPING = 0x03
CMD_ANIMATE = 0x04

# Grayscale command path.
# 0x07: stage one column. Payload = column index byte + 34 brightness bytes.
# 0x08: flush staged columns.
CMD_STAGE_GRAY_COL = 0x07
CMD_FLUSH_GRAY = 0x08

FRAMEWORK_VID = 0x32AC
LED_MATRIX_PID = 0x0020

PANEL_WIDTH = 9
PANEL_HEIGHT = 34

BAUD = 115200
SERIAL_WRITE_TIMEOUT = 2.0
SERIAL_READ_TIMEOUT = 0.05
SERIAL_SETTLE_SECONDS = 0.45

OPEN_PORTS: List[serial.Serial] = []
CLEANING_UP = False

# Last frame cache. Key is port name, value is a 34x9 uint8 panel.
LAST_PANEL_FRAMES: Dict[str, np.ndarray] = {}

try:
    RESAMPLE_DOWNSCALE = Image.Resampling.LANCZOS
    RESAMPLE_PREVIEW = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_DOWNSCALE = Image.LANCZOS
    RESAMPLE_PREVIEW = Image.NEAREST


BAYER_MATRICES = {
    "2x2": np.array([
        [0, 2],
        [3, 1]
    ], dtype=np.float32) / 4.0,

    "4x4": np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.float32) / 16.0,

    "8x8": np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float32) / 64.0
}


def find_led_matrix_ports() -> List[str]:
    devices = []
    for p in serial.tools.list_ports.comports():
        if p.vid == FRAMEWORK_VID and p.pid == LED_MATRIX_PID:
            devices.append(p.device)
    return sorted(devices)


def port_key(port: serial.Serial) -> str:
    return str(getattr(port, "port", id(port)))


def make_packet(cmd: int, payload: bytes = b"") -> bytes:
    return FWK_MAGIC + bytes([cmd]) + payload


def write_exact(
    port: serial.Serial,
    packet: bytes,
    retries: int = 2,
    retry_delay: float = 0.05,
) -> bool:
    for attempt in range(max(1, retries)):
        try:
            if not port or not getattr(port, "is_open", False):
                return False

            written = port.write(packet)
            if written == len(packet):
                return True

            raise serial.SerialTimeoutException(
                f"Short serial write: {written}/{len(packet)} bytes"
            )

        except (serial.SerialTimeoutException, serial.SerialException, OSError):
            try:
                if port and getattr(port, "is_open", False):
                    port.reset_output_buffer()
            except Exception:
                pass

            if attempt + 1 < retries:
                time.sleep(retry_delay)

    return False


def send_cmd(
    port: serial.Serial,
    cmd: int,
    payload: bytes = b"",
    retries: int = 2,
    retry_delay: float = 0.05,
) -> bool:
    packet = make_packet(cmd, payload)
    return write_exact(port, packet, retries=retries, retry_delay=retry_delay)


def set_global_brightness_percent(port: serial.Serial, percent: int, retries: int = 3) -> bool:
    percent = max(0, min(100, int(percent)))
    device_value = int(round(percent * 255 / 100))
    return send_cmd(port, CMD_BRIGHTNESS, bytes([device_value]), retries=retries)


def set_sleeping(port: serial.Serial, sleeping: bool, retries: int = 3) -> bool:
    return send_cmd(port, CMD_SLEEPING, bytes([1 if sleeping else 0]), retries=retries)


def set_animate(port: serial.Serial, animate: bool, retries: int = 3) -> bool:
    return send_cmd(port, CMD_ANIMATE, bytes([1 if animate else 0]), retries=retries)


def blank_gray_panel() -> np.ndarray:
    return np.zeros((PANEL_HEIGHT, PANEL_WIDTH), dtype=np.uint8)


def draw_gray_frame(
    port: serial.Serial,
    panel_gray: np.ndarray,
    retries: int = 1,
    force: bool = False,
    min_change: int = 2,
    serial_burst: bool = False,
) -> bool:
    if panel_gray.shape != (PANEL_HEIGHT, PANEL_WIDTH):
        raise ValueError(
            f"Expected panel shape {(PANEL_HEIGHT, PANEL_WIDTH)}, got {panel_gray.shape}"
        )

    frame = np.clip(panel_gray, 0, 255).astype(np.uint8, copy=False)
    key = port_key(port)
    last = LAST_PANEL_FRAMES.get(key)

    if force or last is None or last.shape != frame.shape:
        changed_columns = list(range(PANEL_WIDTH))
    else:
        threshold = max(1, int(min_change))
        diff = np.abs(frame.astype(np.int16) - last.astype(np.int16))
        changed_columns = [
            int(x)
            for x in np.flatnonzero(diff.max(axis=0) >= threshold)
        ]

    if not changed_columns:
        return True

    if not serial_burst:
        # Safe/default mode: same command cadence as your original code.
        # This is slower than burst mode, but much more likely to work with
        # firmware that reads one command at a time.
        for x in changed_columns:
            payload = bytes([x]) + frame[:, x].tobytes()
            ok = send_cmd(port, CMD_STAGE_GRAY_COL, payload, retries=retries)
            if not ok:
                return False

        ok = send_cmd(port, CMD_FLUSH_GRAY, b"", retries=retries)
        if ok:
            LAST_PANEL_FRAMES[key] = frame.copy()
        return ok

    # Experimental mode: combines multiple protocol packets into one OS write.
    # Only use this if your matrix still updates correctly with --serial-burst.
    burst_parts = []
    for x in changed_columns:
        payload = bytes([x]) + frame[:, x].tobytes()
        burst_parts.append(make_packet(CMD_STAGE_GRAY_COL, payload))
    burst_parts.append(make_packet(CMD_FLUSH_GRAY))

    ok = write_exact(port, b"".join(burst_parts), retries=retries, retry_delay=0.02)
    if ok:
        LAST_PANEL_FRAMES[key] = frame.copy()
    return ok


def clear_port(port: serial.Serial) -> None:
    try:
        if port and getattr(port, "is_open", False):
            draw_gray_frame(
                port,
                blank_gray_panel(),
                retries=3,
                force=True,
                min_change=1,
                serial_burst=False,
            )
            port.flush()
    except Exception:
        pass


def cleanup() -> None:
    global CLEANING_UP

    if CLEANING_UP:
        return

    CLEANING_UP = True

    ports = list(OPEN_PORTS)
    OPEN_PORTS.clear()

    for port in ports:
        try:
            clear_port(port)
            set_animate(port, False, retries=1)
            set_sleeping(port, False, retries=1)
            if port and getattr(port, "is_open", False):
                port.flush()
        except Exception:
            pass

    for port in ports:
        try:
            if port and getattr(port, "is_open", False):
                port.close()
        except Exception:
            pass


def signal_handler(signum, frame) -> None:
    print("\nStopping; clearing matrices...")
    cleanup()
    raise SystemExit(130)


def open_ports(devices: List[str], baud: int = BAUD) -> List[serial.Serial]:
    ports = []

    for device in devices:
        try:
            port = serial.Serial(
                device,
                baud,
                timeout=SERIAL_READ_TIMEOUT,
                write_timeout=SERIAL_WRITE_TIMEOUT,
            )

            ports.append(port)
            OPEN_PORTS.append(port)
            print(f"Opened {device}")

            time.sleep(SERIAL_SETTLE_SECONDS)

            try:
                port.reset_input_buffer()
                port.reset_output_buffer()
            except Exception:
                pass

        except serial.SerialException as e:
            print(f"Could not open {device}: {e}")
            print("On Arch Linux, add yourself to the uucp group:")
            print('  sudo gpasswd -a "$USER" uucp')
            print("Then fully log out and back in.")

    return ports


def initialize_matrix(port: serial.Serial, brightness_percent: int) -> None:
    for _ in range(4):
        ok = True
        ok &= set_sleeping(port, False, retries=4)
        time.sleep(0.03)
        ok &= set_animate(port, False, retries=4)
        time.sleep(0.03)
        ok &= set_global_brightness_percent(port, brightness_percent, retries=4)
        time.sleep(0.03)
        ok &= draw_gray_frame(
            port,
            blank_gray_panel(),
            retries=4,
            force=True,
            min_change=1,
            serial_burst=False,
        )

        if ok:
            try:
                port.flush()
            except Exception:
                pass
            return

        time.sleep(0.25)

    print(f"Warning: startup writes were flaky for {getattr(port, 'port', port)}; continuing.")


def rotate_image(img: Image.Image, rotation: str) -> Image.Image:
    if rotation == "none":
        return img
    if rotation == "cw":
        return img.rotate(-90, expand=True)
    if rotation == "ccw":
        return img.rotate(90, expand=True)
    if rotation == "180":
        return img.rotate(180, expand=True)
    raise ValueError(f"Unknown rotation: {rotation}")


def crop_to_aspect(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    target_aspect = target_width / target_height
    src_width, src_height = img.size
    src_aspect = src_width / src_height

    if src_aspect > target_aspect:
        new_width = int(round(src_height * target_aspect))
        left = max(0, (src_width - new_width) // 2)
        return img.crop((left, 0, left + new_width, src_height))

    if src_aspect < target_aspect:
        new_height = int(round(src_width / target_aspect))
        top = max(0, (src_height - new_height) // 2)
        return img.crop((0, top, src_width, top + new_height))

    return img


def normalize_vizdoom_screen_buffer(buffer: np.ndarray) -> np.ndarray:
    arr = np.asarray(buffer)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Unexpected screen buffer shape: {arr.shape}")

    # CHW -> HWC
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    if arr.shape[-1] != 3:
        raise ValueError(f"Unexpected RGB buffer shape: {arr.shape}")

    return arr.astype(np.uint8, copy=False)


def cie1931_correction(arr: np.ndarray) -> np.ndarray:
    """
    Apply CIE 1931 lightness-to-luminance curve mapping.
    Converts perceived linear brightness 0..255 to physical LED PWM duty cycle 0..255.
    """
    L = arr * (100.0 / 255.0)
    Y = np.where(
        L <= 8.0,
        L / 903.3,
        np.power((L + 16.0) / 116.0, 3.0)
    )
    return Y * 255.0


def apply_levels(arr: np.ndarray, levels: int) -> np.ndarray:
    levels = max(2, min(256, int(levels)))

    if levels >= 256:
        return arr

    step = 255.0 / float(levels - 1)
    return np.round(arr / step) * step


def apply_ordered_dither(arr: np.ndarray, mode: str, levels: int) -> np.ndarray:
    if mode not in BAYER_MATRICES:
        return apply_levels(arr, levels)

    mat = BAYER_MATRICES[mode]
    h, w = arr.shape
    mh, mw = mat.shape

    tmat = np.tile(mat, (int(np.ceil(h / mh)), int(np.ceil(w / mw))))[:h, :w]

    levels = max(2, min(256, int(levels)))
    step = 255.0 / float(levels - 1)
    offset = (tmat - 0.5) * step

    dithered = np.clip(arr + offset, 0, 255)
    return np.round(dithered / step) * step


def doom_frame_to_gray_canvas(
    rgb: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    game_rotate: str,
    fit_mode: str,
    autocontrast: bool,
    autocontrast_cutoff: float,
    equalize: bool,
    contrast: float,
    sharpen: float,
    invert: bool,
    gamma: float,
    levels: int,
    cie: bool,
    dither_mode: str,
    edge_boost: float,
) -> np.ndarray:
    img = Image.fromarray(rgb, mode="RGB")
    img = rotate_image(img, game_rotate)

    if fit_mode == "crop":
        img = crop_to_aspect(img, canvas_width, canvas_height)

    gray = img.convert("L")
    gray = gray.resize((canvas_width, canvas_height), RESAMPLE_DOWNSCALE)

    if equalize:
        gray = ImageOps.equalize(gray)

    if autocontrast:
        gray = ImageOps.autocontrast(gray, cutoff=max(0.0, float(autocontrast_cutoff)))

    contrast = max(0.1, float(contrast))
    if abs(contrast - 1.0) > 0.001:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)

    sharpen = max(0.0, float(sharpen))
    if sharpen > 0.001:
        percent = int(round(80 + sharpen * 140))
        gray = gray.filter(
            ImageFilter.UnsharpMask(
                radius=1.0,
                percent=percent,
                threshold=1,
            )
        )

    arr = np.asarray(gray).astype(np.float32)

    if edge_boost > 0.001:
        temp_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        edges = temp_img.filter(ImageFilter.FIND_EDGES)
        arr_edges = np.asarray(edges).astype(np.float32)
        arr = arr + (arr_edges * edge_boost)

    if invert:
        arr = 255.0 - arr

    if cie:
        arr = cie1931_correction(arr)
    else:
        gamma = max(0.05, float(gamma))
        if abs(gamma - 1.0) > 0.001:
            arr = 255.0 * np.power(np.clip(arr / 255.0, 0.0, 1.0), gamma)

    if dither_mode != "none":
        arr = apply_ordered_dither(arr, dither_mode, levels)
    else:
        arr = apply_levels(arr, levels)

    return np.clip(arr, 0, 255).astype(np.uint8)


def transform_canvas(canvas_gray: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    out = canvas_gray

    if flip_x:
        out = np.fliplr(out)

    if flip_y:
        out = np.flipud(out)

    return out


def send_canvas_to_matrices(
    ports: List[serial.Serial],
    canvas_gray: np.ndarray,
    flip_x: bool,
    flip_y: bool,
    reverse_ports: bool,
    min_change: int,
    serial_executor: Optional[ThreadPoolExecutor] = None,
    serial_burst: bool = False,
) -> None:
    canvas_gray = transform_canvas(canvas_gray, flip_x=flip_x, flip_y=flip_y)

    send_ports = list(ports)
    if reverse_ports:
        send_ports.reverse()

    expected_shape = (PANEL_HEIGHT, PANEL_WIDTH * len(send_ports))
    if canvas_gray.shape != expected_shape:
        raise ValueError(f"Expected canvas shape {expected_shape}, got {canvas_gray.shape}")

    jobs = []
    for i, port in enumerate(send_ports):
        x0 = i * PANEL_WIDTH
        x1 = x0 + PANEL_WIDTH
        panel = canvas_gray[:, x0:x1].copy()
        jobs.append((port, panel))

    if serial_executor is not None and len(jobs) > 1:
        futures = [
            serial_executor.submit(
                draw_gray_frame,
                port,
                panel,
                1,
                False,
                min_change,
                serial_burst,
            )
            for port, panel in jobs
        ]
        for future in futures:
            future.result()
    else:
        for port, panel in jobs:
            draw_gray_frame(
                port,
                panel,
                retries=1,
                force=False,
                min_change=min_change,
                serial_burst=serial_burst,
            )


def call_if_exists(obj, method_name: str, *method_args) -> bool:
    method = getattr(obj, method_name, None)

    if callable(method):
        method(*method_args)
        return True

    return False


def find_builtin_vizdoom_basic_cfg(vzd) -> Optional[str]:
    candidates = []

    scenarios_path = getattr(vzd, "scenarios_path", None)
    if scenarios_path:
        if callable(scenarios_path):
            try:
                scenarios_path = scenarios_path()
            except TypeError:
                scenarios_path = None

        if scenarios_path:
            candidates.append(os.path.join(str(scenarios_path), "basic.cfg"))

    candidates.append(os.path.join(os.path.dirname(vzd.__file__), "scenarios", "basic.cfg"))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None


def guess_map_from_iwad(iwad_path: Optional[str]) -> Optional[str]:
    if not iwad_path:
        return None

    name = os.path.basename(iwad_path).lower()

    doom2_names = ["doom2", "freedoom2", "plutonia", "tnt", "phase2"]
    doom1_names = ["doom.wad", "doom1", "freedoom1", "ultimate", "phase1"]

    if any(token in name for token in doom2_names):
        return "map01"

    if any(token in name for token in doom1_names):
        return "e1m1"

    return None


def init_doom(args):
    import vizdoom as vzd

    game = vzd.DoomGame()
    loaded_config = False

    if args.config:
        game.load_config(args.config)
        loaded_config = True

    elif not args.iwad:
        basic_cfg = find_builtin_vizdoom_basic_cfg(vzd)
        if basic_cfg:
            print(f"No --iwad supplied; using ViZDoom demo config: {basic_cfg}")
            game.load_config(basic_cfg)
            loaded_config = True

    if args.iwad:
        game.set_doom_game_path(args.iwad)

    if args.pwad:
        game.set_doom_scenario_path(args.pwad)

    chosen_map = args.map or guess_map_from_iwad(args.iwad)
    if chosen_map:
        print(f"Starting map: {chosen_map}")
        game.set_doom_map(chosen_map)

    resolution = getattr(vzd.ScreenResolution, "RES_320X240", None)
    if resolution is None:
        resolution = getattr(vzd.ScreenResolution, "RES_160X120")

    game.set_screen_resolution(resolution)
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    call_if_exists(game, "set_render_hud", args.hud)
    call_if_exists(game, "set_render_minimal_hud", False)
    call_if_exists(game, "set_render_crosshair", False)
    call_if_exists(game, "set_render_weapon", True)
    call_if_exists(game, "set_render_decals", False)
    call_if_exists(game, "set_render_particles", False)
    call_if_exists(game, "set_render_effects", True)
    call_if_exists(game, "set_render_corpses", True)
    call_if_exists(game, "set_render_messages", False)

    button_names = [
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
        "USE",
        "SPEED",
    ]

    buttons_by_name = {
        name: getattr(vzd.Button, name)
        for name in button_names
        if hasattr(vzd.Button, name)
    }

    buttons = list(buttons_by_name.values())

    game.set_available_buttons(buttons)
    call_if_exists(game, "set_window_visible", args.show_doom_window)
    game.set_mode(vzd.Mode.PLAYER)

    if not loaded_config and not args.iwad:
        raise RuntimeError(
            "No Doom IWAD/config found. Pass --iwad /path/to/DOOM.WAD "
            "or --config /path/to/vizdoom.cfg"
        )

    game.init()
    game.new_episode()

    return game, vzd, buttons, buttons_by_name


def pygame_action_from_keyboard(
    pygame,
    keys,
    mouse_buttons,
    buttons: List[object],
    buttons_by_name: Dict[str, object],
    wasd_strafes: bool,
) -> List[int]:
    forward = keys[pygame.K_w] or keys[pygame.K_UP]
    backward = keys[pygame.K_s] or keys[pygame.K_DOWN]

    if wasd_strafes:
        turn_left = keys[pygame.K_LEFT] or keys[pygame.K_q]
        turn_right = keys[pygame.K_RIGHT] or keys[pygame.K_e]
        strafe_left = keys[pygame.K_a]
        strafe_right = keys[pygame.K_d]
    else:
        turn_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        turn_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        strafe_left = keys[pygame.K_q]
        strafe_right = keys[pygame.K_e]

    attack = (
        keys[pygame.K_LCTRL]
        or keys[pygame.K_RCTRL]
        or keys[pygame.K_z]
        or keys[pygame.K_f]
        or mouse_buttons[0]
    )

    use = keys[pygame.K_SPACE] or keys[pygame.K_RETURN]
    speed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    pressed_by_name = {
        "MOVE_FORWARD": forward,
        "MOVE_BACKWARD": backward,
        "TURN_LEFT": turn_left,
        "TURN_RIGHT": turn_right,
        "MOVE_LEFT": strafe_left,
        "MOVE_RIGHT": strafe_right,
        "ATTACK": attack,
        "USE": use,
        "SPEED": speed,
    }

    values_by_button = {
        button: pressed_by_name.get(name, False)
        for name, button in buttons_by_name.items()
    }

    return [1 if values_by_button.get(button, False) else 0 for button in buttons]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Doom on Framework 16 LED Matrix modules with human-perceptual visual enhancements."
    )

    parser.add_argument(
        "--port",
        action="append",
        help="Serial port to use. Example: --port /dev/ttyACM0"
    )

    parser.add_argument("--list", action="store_true", help="List detected LED Matrix modules and exit.")

    parser.add_argument(
        "--brightness",
        type=int,
        default=100,
        help="Global max brightness percent, 0..100. Default: 100.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=14.0,
        help="LED update FPS. Default: 14.",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD,
        help="Serial baud rate to request. Default: 115200.",
    )

    parser.add_argument(
        "--serial-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel serial writers. Start with 1. "
            "Use 2 for two matrix modules after confirming LEDs work. Default: 1."
        ),
    )

    parser.add_argument(
        "--serial-burst",
        action="store_true",
        help=(
            "Experimental: combine all staged-column packets and the flush into "
            "one write per panel. Faster if supported; disable if LEDs stay blank."
        ),
    )

    parser.add_argument(
        "--cie",
        action="store_true",
        default=True,
        help="Enable CIE 1931 lightness curve. Default: True.",
    )

    parser.add_argument(
        "--no-cie",
        action="store_false",
        dest="cie",
        help="Disable CIE 1931 mapping and use simple power-law gamma instead.",
    )

    parser.add_argument(
        "--dither",
        choices=["none", "2x2", "4x4", "8x8"],
        default="4x4",
        help="Apply spatial ordered Bayer dithering. Default: 4x4.",
    )

    parser.add_argument(
        "--edge-boost",
        type=float,
        default=0.20,
        help="Outline enhancement factor, 0.0 to 1.0. Default: 0.20.",
    )

    parser.add_argument(
        "--equalize",
        action="store_true",
        help="Perform dynamic global histogram equalization.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Standard gamma correction. Used only if --no-cie is passed. Default: 1.0.",
    )

    parser.add_argument(
        "--contrast",
        type=float,
        default=1.45,
        help="Contrast boost before dither. Default: 1.45.",
    )

    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.65,
        help="Edge sharpening amount. Default: 0.65.",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=64,
        help="Grayscale target levels. Default: 64.",
    )

    parser.add_argument(
        "--min-change",
        type=int,
        default=2,
        help="Only resend columns with at least this much brightness change. Default: 2.",
    )

    parser.add_argument(
        "--fit-mode",
        choices=["crop", "squash"],
        default="crop",
        help="crop keeps Doom readable; squash shows the whole frame. Default: crop.",
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert grayscale output.",
    )

    parser.add_argument(
        "--no-autocontrast",
        action="store_true",
        help="Disable per-frame autocontrast.",
    )

    parser.add_argument(
        "--autocontrast-cutoff",
        type=float,
        default=1.0,
        help="Autocontrast cutoff percent. Default: 1.0.",
    )

    parser.add_argument(
        "--game-rotate",
        choices=["none", "cw", "ccw", "180"],
        default="cw",
        help="Rotate Doom before resizing to 18x34. Default: cw.",
    )

    parser.add_argument("--flip-x", action="store_true", help="Flip the final image horizontally.")
    parser.add_argument("--flip-y", action="store_true", help="Flip the final image vertically.")
    parser.add_argument("--reverse-ports", action="store_true", help="Swap left/right panel order.")

    parser.add_argument(
        "--preview-scale",
        type=int,
        default=7,
        help="Pygame preview scale. Set to 0 to show only Doom preview. Default: 7.",
    )

    parser.add_argument("--wasd-strafes", action="store_true", help="Use A/D for strafe; Q/E become turn.")

    parser.add_argument("--iwad", help="Path to Doom IWAD, e.g. DOOM.WAD.")
    parser.add_argument("--pwad", help="Optional PWAD / scenario WAD path.")
    parser.add_argument("--config", help="Optional ViZDoom config path.")
    parser.add_argument("--map", default=None, help="Map to start on. Examples: e1m1 or map01.")
    parser.add_argument("--hud", action="store_true", help="Render Doom HUD before downscaling.")
    parser.add_argument("--show-doom-window", action="store_true", help="Also show ViZDoom's own window.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)

    devices = args.port or find_led_matrix_ports()

    if args.list:
        if devices:
            print("Detected Framework LED Matrix modules:")
            for d in devices:
                print(f"  {d}")
        else:
            print("No Framework LED Matrix modules detected.")
        return 0

    if not devices:
        print("No Framework LED Matrix modules detected.")
        print("Try:")
        print("  python doom.py --port /dev/ttyACM0 --port /dev/ttyACM1 --iwad DOOM.WAD")
        return 1

    print(f"Found {len(devices)} LED Matrix module(s): {', '.join(devices)}")
    print(f"Output resolution: {PANEL_WIDTH * len(devices)}x{PANEL_HEIGHT} grayscale")

    ports = open_ports(devices, baud=max(9600, int(args.baud)))
    if not ports:
        print("No usable serial ports opened.")
        return 1

    brightness_percent = max(0, min(100, args.brightness))

    for port in ports:
        initialize_matrix(port, brightness_percent)

    canvas_width = PANEL_WIDTH * len(ports)
    canvas_height = PANEL_HEIGHT

    import pygame

    pygame.init()
    pygame.display.set_caption("Doom Dashboard | Game (Left) | LED Matrix (Right)")

    doom_w, doom_h = 320, 240
    preview_scale = max(0, args.preview_scale)
    led_w = canvas_width * preview_scale
    led_h = canvas_height * preview_scale
    padding = 20

    if preview_scale > 0:
        surface = pygame.display.set_mode((doom_w + padding + led_w, max(doom_h, led_h)))
    else:
        surface = pygame.display.set_mode((doom_w, doom_h))

    clock = pygame.time.Clock()
    game = None
    serial_executor = None

    serial_workers = max(1, min(int(args.serial_workers), len(ports)))
    if serial_workers > 1:
        serial_executor = ThreadPoolExecutor(max_workers=serial_workers)
        print(f"Parallel serial writers: {serial_workers}")

    print(f"Serial packet mode: {'burst' if args.serial_burst else 'safe'}")

    try:
        game, vzd, buttons, buttons_by_name = init_doom(args)

        print("\nControls:")
        print("  W/S/A/D or Arrow keys for movement.")
        print("  Space/Enter to use. Ctrl/Z/F to fire.")
        print("  Esc to quit. R to restart.")

        print("\nOptimizations active:")
        print(f"  * CIE 1931 Curve: {args.cie}")
        print(f"  * Edge Outlining: {args.edge_boost > 0.0} factor={args.edge_boost}")
        print(f"  * Dither Mode: {args.dither} levels={args.levels}")
        print(f"  * Histogram Equalization: {args.equalize}")
        print(f"  * Min Change: {args.min_change}")
        print(f"  * Preview Scale: {preview_scale}")

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.new_episode()

            if game.is_episode_finished():
                game.new_episode()

            keys = pygame.key.get_pressed()
            mouse_buttons = pygame.mouse.get_pressed()

            action = pygame_action_from_keyboard(
                pygame=pygame,
                keys=keys,
                mouse_buttons=mouse_buttons,
                buttons=buttons,
                buttons_by_name=buttons_by_name,
                wasd_strafes=args.wasd_strafes,
            )

            game.make_action(action, 1)

            state = game.get_state()
            if state is None:
                clock.tick(args.fps)
                continue

            rgb = normalize_vizdoom_screen_buffer(state.screen_buffer)
            curr_doom_h, curr_doom_w = rgb.shape[0], rgb.shape[1]

            canvas_gray = doom_frame_to_gray_canvas(
                rgb=rgb,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                game_rotate=args.game_rotate,
                fit_mode=args.fit_mode,
                autocontrast=not args.no_autocontrast,
                autocontrast_cutoff=args.autocontrast_cutoff,
                equalize=args.equalize,
                contrast=args.contrast,
                sharpen=args.sharpen,
                invert=args.invert,
                gamma=args.gamma,
                levels=args.levels,
                cie=args.cie,
                dither_mode=args.dither,
                edge_boost=args.edge_boost,
            )

            send_canvas_to_matrices(
                ports=ports,
                canvas_gray=canvas_gray,
                flip_x=args.flip_x,
                flip_y=args.flip_y,
                reverse_ports=args.reverse_ports,
                min_change=args.min_change,
                serial_executor=serial_executor,
                serial_burst=args.serial_burst,
            )

            curr_led_w = canvas_width * preview_scale
            curr_led_h = canvas_height * preview_scale

            expected_w = curr_doom_w + (padding + curr_led_w if preview_scale > 0 else 0)
            expected_h = max(curr_doom_h, curr_led_h) if preview_scale > 0 else curr_doom_h

            if surface.get_width() != expected_w or surface.get_height() != expected_h:
                surface = pygame.display.set_mode((expected_w, expected_h))

            surface.fill((20, 20, 20))

            doom_surf = pygame.image.frombytes(rgb.tobytes(), (curr_doom_w, curr_doom_h), "RGB")
            doom_y = (surface.get_height() - curr_doom_h) // 2
            surface.blit(doom_surf, (0, doom_y))

            if preview_scale > 0:
                preview_gray = transform_canvas(canvas_gray, args.flip_x, args.flip_y)
                img = Image.fromarray(preview_gray, mode="L")
                img_resized = img.resize((curr_led_w, curr_led_h), RESAMPLE_PREVIEW).convert("RGB")
                led_surf = pygame.image.frombytes(img_resized.tobytes(), img_resized.size, "RGB")

                led_y = (surface.get_height() - curr_led_h) // 2
                surface.blit(led_surf, (curr_doom_w + padding, led_y))

                pygame.draw.line(
                    surface,
                    (60, 60, 60),
                    (curr_doom_w + padding // 2, 0),
                    (curr_doom_w + padding // 2, surface.get_height()),
                    1,
                )

            pygame.display.flip()
            clock.tick(args.fps)

    finally:
        if serial_executor is not None:
            try:
                serial_executor.shutdown(wait=True)
            except Exception:
                pass

        print("\nClearing matrices...")
        cleanup()

        if game is not None:
            try:
                game.close()
            except Exception:
                pass

        try:
            pygame.quit()
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
