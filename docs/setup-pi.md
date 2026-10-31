# Setting up the Pi

1. Flash **Raspberry Pi OS (64-bit)** with Raspberry Pi Imager. Set hostname `pokedex`, your
   user, Wi-Fi and SSH in the Imager settings.
2. Connect the camera (Pi 5 needs the narrow 22-pin ribbon) and screen, then boot and check:

   ```bash
   rpicam-hello -t 5s
   ```

3. Apply [`deploy/config.txt.snippet`](../deploy/config.txt.snippet) to `/boot/firmware/config.txt`
   and reboot.
4. Install and run:

   ```bash
   sudo apt update && sudo apt install -y git python3-picamera2 python3-pygame espeak-ng
   git clone https://github.com/<you>/pokedex.git && cd pokedex
   python -m venv --system-site-packages .venv && source .venv/bin/activate
   pip install -r requirements-pi.txt
   python scripts/fetch_pokeapi.py
   python -m pokedex --fullscreen --gpio
   ```

5. Speaker check: `speaker-test -t wav -c 2` and `espeak-ng "Pokédex online"`.
6. Allow the shutdown button to work without a password:

   ```bash
   echo "$USER ALL=(root) NOPASSWD: /usr/sbin/shutdown" | sudo tee /etc/sudoers.d/pokedex
   ```

7. Auto-start: install [`deploy/pokedex.service`](../deploy/pokedex.service) (edit `User` and paths).
8. Check temperatures with the shell closed while scanning repeatedly:
   `watch -n1 vcgencmd measure_temp`. Aim for under 80 °C.

## Optional: nicer voice with Piper

Install [Piper](https://github.com/rhasspy/piper), download a voice `.onnx`, then set in
`config.toml`:

```toml
[audio]
tts = "piper"
piper_voice = "/home/pi/voices/en_US-amy-medium.onnx"
```
