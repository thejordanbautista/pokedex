# Real-Life Pokédex

A handheld Pokédex built on a Raspberry Pi 5: point the camera, press **Scan**, and it shows the
entry and reads it aloud. Two recognition modes:

- **Pokémon mode** — recognizes Pokémon in cards, plushies, figures and screens.
- **World mode** — recognizes real animals and objects and shows a Pokédex-style entry you write.

The same code runs on a laptop (webcam or a folder of test images, keyboard controls), so you can
build the software before the hardware arrives.

## Quick start (laptop)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_pokeapi.py   # Gen 1 stats, sprites, cries (~2 min)
python -m pokedex                 # webcam if found, else images in samples/
python -m pokedex --camera folder # cycle through samples/ instead of a webcam
```

Without a model in `models/`, a mock classifier cycles through the dex so you can work on the UI.
To make **World mode** recognize real animals and objects right away:

```bash
python scripts/fetch_world_model.py
python -m pokedex --mode world      # point it at a mug, a banana, a dog...
```

It runs with OpenCV, which is already installed. `pip install ai-edge-litert` makes it faster where it
works (not on macOS with Python 3.9). On the Pi, use the smaller model:

```bash
python scripts/fetch_world_model.py --quantized && pip install ai-edge-litert
```

**Webcam not showing?** Run `python -m pokedex.camera`. It tests each camera slot, saves a snapshot
from each into `camera-test/`, and says why any slot failed. On macOS, allow your terminal in
System Settings > Privacy & Security > Camera, then quit and reopen the terminal. If slot 0 is your
iPhone (Continuity Camera), use `python -m pokedex --camera-index 1`, or press **C** in the app.

| Key | Action |
| --- | --- |
| Space / Enter | Scan |
| Arrow keys | D-pad |
| Z | A (select) |
| X / Esc | B (back) |
| M | Switch Pokémon / World mode |
| C | Try the next camera |
| Q | Quit |

## On the Raspberry Pi

```bash
sudo apt install -y python3-picamera2 python3-pygame espeak-ng
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
python scripts/fetch_pokeapi.py          # Gen 1 stats, sprites and cries (personal use)
python -m pokedex --fullscreen --gpio
```

Auto-start on boot: see [`deploy/`](deploy/). Wiring and pins: [`docs/wiring.md`](docs/wiring.md).

## Models

Drop TensorFlow Lite models into `models/`, and install the runtime once you have one:
`pip install ai-edge-litert`.

| File | What |
| --- | --- |
| `pokemon.tflite` + `pokemon_labels.txt` | Pokémon classifier |
| `world.tflite` + `world_labels.txt` | Real-world classifier (e.g. ImageNet MobileNet) |

Photos taken with the device's own camera matter more than model size. The training loop:

```bash
# 1. On the Pokédex: capture photos per class (Space = burst of 10, Q = done)
python -m pokedex.capture Pikachu
python -m pokedex.capture background --auto 100   # random scenes = "nothing here"

# 2. On a laptop or Colab (not the Pi): train and export int8 TFLite
python scripts/train_classifier.py photos/ --out models/pokemon

# 3. Back on the Pi: check accuracy, scan time and a suggested threshold
#    (use a fresh set of photos the model hasn't seen)
python -m pokedex.evaluate test-photos/ --mode pokemon
```

Phase 3 is done when accuracy is at least 85% and the median scan time is under 1 s on the Pi.

## Layout

```
pokedex/            app package (camera, classifier, dex data, audio, controls, UI)
data/               dex entries (world.json is hand-written; pokemon.json is generated)
models/             .tflite models + labels (not committed)
assets/             sprites and cries (generated, not committed)
samples/            test images for the folder camera
scripts/            data fetch + training
deploy/             systemd service and config.txt snippet
hardware/           CAD and print files for the shell
docs/               build notes and wiring
tests/
```

## Roadmap

- [ ] 1. Bench setup — Pi OS 64-bit, camera + screen preview working
- [ ] 2. App runs on the Pi with the folder camera and mock classifier
- [ ] 3. Pokémon model — train on own photos, int8 TFLite, measure accuracy and scan time
- [ ] 4. World mode — pretrained object model + custom entries
- [ ] 5. Buttons and I2S audio, text-to-speech
- [ ] 6. Battery power, battery indicator, safe shutdown
- [ ] 7. Shell v1 — CAD around the measured stack, fit-test prints
- [ ] 8. Assembly — inserts, hinge, lens dome and LEDs, thermals
- [ ] 9. Polish — boot splash, sounds, seen/caught log

## Legal

The code is MIT-licensed. Pokémon names, sprites, cries and the Pokédex design belong to
Nintendo / Game Freak / The Pokémon Company. `scripts/fetch_pokeapi.py` downloads them for personal
use only; they are git-ignored and should not be redistributed. Don't sell the device or the data.
