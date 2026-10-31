# Shell (CAD and print files)

Put CAD sources and print files here as the shell takes shape:

```
hardware/
  cad/        Fusion 360 / Onshape exports, FreeCAD files, STEP models of parts
  stl/        print-ready files
  photos/     fit tests and assembly
```

## Design order

1. Measure the real parts (or download STEP models): Pi 5 + Active Cooler, battery board and cell,
   screen, camera, speaker, buttons.
2. Block out the stack, then shell around it: 2–2.5 mm walls, a lip between halves, bosses for
   M2.5/M3 heat-set inserts.
3. Leave room for the camera ribbon, USB-C charge port, power switch, speaker grille and vents
   (intake under the Pi, exhaust by the fan).
4. Print the screen bezel and button panel alone first to check fit (0.2–0.4 mm clearance).

## Print settings that work

PETG (or PLA if it stays cool), 0.2 mm layers, 3 walls, 15–20 % infill, faces printed flat.
Hinge: print-in-place, or a 2–3 mm steel pin through printed knuckles.

## Parts to measure (fill in)

| Part | W × H × D (mm) | Notes |
| --- | --- | --- |
| Pi 5 + Active Cooler | 85 × 56 × ~20 | |
| Battery board + cell | | |
| Screen (module) | | |
| Screen (active area) | | |
| Camera board | 25 × 24 | |
| Speaker | | |
