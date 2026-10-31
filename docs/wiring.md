# Wiring

Power flows one way: battery cell → battery board (Pi 5 rated, 5 V 5 A) → Pi.
Camera and display use ribbon cables; everything else is a few GPIO wires.

Buttons connect between the GPIO pin and any GND pin; internal pull-ups are enabled in software.
Pins are BCM numbers and can be changed in `config.toml`.

| Function | GPIO | Header pin | Notes |
| --- | --- | --- | --- |
| I2S amp BCLK | 18 | 12 | `dtoverlay=max98357a` |
| I2S amp LRCLK | 19 | 35 | |
| I2S amp DIN | 21 | 40 | Amp VIN to 5 V (pin 2), GND to pin 6 |
| D-pad up | 5 | 29 | |
| D-pad down | 6 | 31 | |
| D-pad left | 13 | 33 | |
| D-pad right | 26 | 37 | |
| A | 16 | 36 | |
| B (hold 3 s = shutdown) | 17 | 11 | |
| Scan | 27 | 13 | |
| Lens LEDs | 22 | 15 | Through an NPN transistor or MOSFET, not straight from the pin |

If you use an SPI screen instead of DSI, check which pins it takes (usually GPIO 8–11 and 24–25)
and move anything that clashes.

## Lens LED driver

```
GPIO22 ──[1 kΩ]── base   2N2222 / BC337
                  emitter ── GND
5 V ──[LED]──[68 Ω]── collector     (one resistor per LED; size it for your LED's current)
```
