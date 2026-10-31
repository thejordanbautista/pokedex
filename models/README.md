# Models

Put `.tflite` models and their label files here (they're git-ignored):

| File | Used for |
| --- | --- |
| `pokemon.tflite`, `pokemon_labels.txt` | Pokémon mode |
| `world.tflite`, `world_labels.txt` | World mode |

Label files have one class per line, in the model's output order (`0 tabby` style is fine too).
Labels must match an entry's `name` or one of its `labels` in `data/*.json`.

- **Pokémon:** train with `scripts/train_classifier.py` on your own photos.
- **World:** any ImageNet MobileNet `.tflite` (float or uint8) with its labels file works;
  `data/world.json` already maps common ImageNet labels to entries.

With no model present, the app uses a mock classifier that cycles through the dex.
