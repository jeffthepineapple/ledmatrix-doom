# Doom Framework LED Matrix

Run Doom on Framework Laptop 16 LED Matrix modules.

## Install

Install the Python packages:

```sh
python -m pip install --user pyserial pygame pillow numpy vizdoom
```

If your system needs it, use:

```sh
python -m pip install --user --break-system-packages pyserial pygame pillow numpy vizdoom
```

## Run

You must provide your own Doom WAD file. This project does not include `DOOM.WAD`.

Basic run:

```sh
python doom.py --iwad /path/to/DOOM.WAD
```

Best settings found so far:

```sh
python doom.py --iwad /path/to/DOOM.WAD --fps 60 --serial-workers 2 --preview-scale 7 --no-cie --gamma 2.2 --brightness 65 --contrast 1.1
```

If the left and right LED panels are swapped, add `--reverse-ports`:

```sh
python doom.py --iwad /path/to/DOOM.WAD --fps 60 --serial-workers 2 --preview-scale 7 --no-cie --gamma 2.2 --brightness 65 --contrast 1.1 --reverse-ports
```

List detected LED Matrix modules:

```sh
python doom.py --list
```

## Options

- `--iwad PATH` - Path to Doom IWAD, like `DOOM.WAD`.
- `--pwad PATH` - Optional PWAD file.
- `--config PATH` - Optional ViZDoom config file.
- `--map MAP` - Start on a map, like `e1m1` or `map01`.
- `--port PORT` - Serial port to use. Can be used more than once.
- `--list` - List detected LED Matrix modules and exit.
- `--brightness N` - Brightness percent, `0` to `100`. Default: `100`.
- `--fps N` - LED update FPS. Default: `14`.
- `--baud N` - Serial baud rate. Default: `115200`.
- `--serial-workers N` - Number of parallel serial writers. Default: `1`.
- `--serial-burst` - Experimental faster serial writing.
- `--preview-scale N` - Pygame preview scale. Use `0` to hide the LED preview. Default: `7`.
- `--reverse-ports` - Swap left and right LED panels.
- `--flip-x` - Flip the final image horizontally.
- `--flip-y` - Flip the final image vertically.
- `--game-rotate none|cw|ccw|180` - Rotate Doom before resizing. Default: `cw`.
- `--fit-mode crop|squash` - Crop or squash the Doom image. Default: `crop`.
- `--hud` - Render the Doom HUD.
- `--show-doom-window` - Also show ViZDoom's own window.
- `--wasd-strafes` - Use A/D to strafe, with Q/E for turning.
- `--cie` - Enable CIE brightness mapping. Default: on.
- `--no-cie` - Disable CIE brightness mapping.
- `--gamma N` - Gamma correction when using `--no-cie`. Default: `1.0`.
- `--dither none|2x2|4x4|8x8` - Dithering mode. Default: `4x4`.
- `--levels N` - Grayscale levels. Default: `64`.
- `--contrast N` - Contrast boost. Default: `1.45`.
- `--sharpen N` - Sharpen amount. Default: `0.65`.
- `--edge-boost N` - Outline enhancement. Default: `0.20`.
- `--equalize` - Use histogram equalization.
- `--invert` - Invert grayscale output.
- `--no-autocontrast` - Disable per-frame autocontrast.
- `--autocontrast-cutoff N` - Autocontrast cutoff percent. Default: `1.0`.

## Screen mirror to the keyboard

Moved to [`fw16-screen-mirror`](https://github.com/jeffthepineapple/fw16-screen-mirror):
streams the display to the Framework 16 keyboard backlight over QMK raw HID,
plus the custom firmware (`0xF0` host-control protocol) that implements it.
Unrelated to the LED Matrix modules `doom.py` drives.

## Controls

Use the game window to play. Press `Ctrl+C` in the terminal to stop and clear the LED matrices.
