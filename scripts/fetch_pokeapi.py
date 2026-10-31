#!/usr/bin/env python3
"""Download Pokémon data, sprites and cries from PokéAPI into data/ and assets/.

Run once on a machine with internet (laptop or Pi). Everything it writes is git-ignored:
the names, sprites and cries are owned by Nintendo / Game Freak / The Pokémon Company and are
for personal use only.

    python scripts/fetch_pokeapi.py                # Gen 1 (1-151)
    python scripts/fetch_pokeapi.py --last 251     # Gen 1-2
    python scripts/fetch_pokeapi.py --no-media     # data only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://pokeapi.co/api/v2"
UA = {"User-Agent": "real-life-pokedex (personal project)"}


def get_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {url} ({e})", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def download(url: str | None, dest: Path) -> bool:
    if not url:
        return False
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:
        print(f"  could not fetch {url}: {e}", file=sys.stderr)
        return False


def english(entries: list[dict], key: str, version_pref=("red", "blue", "yellow")) -> str:
    en = [e for e in entries if e["language"]["name"] == "en"]
    if not en:
        return ""
    if "version" in en[0]:
        en.sort(key=lambda e: version_pref.index(e["version"]["name"]) if e["version"]["name"] in version_pref else 99)
    text = en[0][key]
    return re.sub(r"\s+", " ", text.replace("\f", " ")).strip()


def chain_names(node: dict) -> list[str]:
    """Flatten an evolution chain, depth-first (branches like Eevee's are listed in order)."""
    names = [node["species"]["name"].title()]
    for child in node["evolves_to"]:
        names += chain_names(child)
    return names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--last", type=int, default=151)
    p.add_argument("--no-media", action="store_true")
    args = p.parse_args()

    assets = ROOT / "assets"
    chains: dict[str, list[str]] = {}
    entries = []
    for n in range(args.first, args.last + 1):
        mon = get_json(f"{API}/pokemon/{n}")
        species = get_json(mon["species"]["url"])
        chain_url = species["evolution_chain"]["url"]
        if chain_url not in chains:
            chains[chain_url] = chain_names(get_json(chain_url)["chain"])

        name = english(species["names"], "name") or mon["name"].title()
        sprite = cry = None
        if not args.no_media:
            sprites = mon["sprites"]
            gen1 = sprites.get("versions", {}).get("generation-i", {}).get("red-blue", {}).get("front_transparent")
            if download(sprites.get("front_default") or gen1, assets / "sprites" / f"{n:03d}.png"):
                sprite = f"sprites/{n:03d}.png"
            cries = mon.get("cries") or {}
            if download(cries.get("legacy") or cries.get("latest"), assets / "cries" / f"{n:03d}.ogg"):
                cry = f"cries/{n:03d}.ogg"

        entries.append({
            "number": n,
            "name": name,
            "category": english(species["genera"], "genus"),
            "types": [t["type"]["name"] for t in sorted(mon["types"], key=lambda t: t["slot"])],
            "height_m": mon["height"] / 10,
            "weight_kg": mon["weight"] / 10,
            "description": english(species["flavor_text_entries"], "flavor_text"),
            "evolutions": chains[chain_url],
            "labels": [mon["name"], name],
            "sprite": sprite,
            "cry": cry,
        })
        print(f"{n:03d} {name}")

    data = ROOT / "data" / "pokemon.json"
    data.write_text(json.dumps({"entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    labels = ROOT / "models" / "pokemon_labels.example.txt"
    labels.write_text("\n".join(e["name"] for e in entries) + "\n", encoding="utf-8")
    print(f"\nWrote {len(entries)} entries to {data.relative_to(ROOT)}")
    print(f"Label list for training: {labels.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
