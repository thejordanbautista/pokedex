// Pokémon data from PokéAPI, fetched by the phone itself and cached on the device.
// Nothing here is bundled into the site: names, text and artwork belong to Nintendo /
// Game Freak / The Pokémon Company and are only looked up for personal use.

const API = "https://pokeapi.co/api/v2";

const title = (s) => s.split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
const english = (list, key) => (list || []).find((x) => x.language?.name === "en")?.[key] || "";
const clean = (t) => t.replace(/[\f\n\r­]+/g, " ").replace(/\s+/g, " ").trim();

function chainNames(node) {
  return [title(node.species.name), ...node.evolves_to.flatMap(chainNames)];
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status === 404 ? "not_found" : `PokéAPI error ${res.status}`);
  return res.json();
}

// All species names, for the search box. ~1000 names, cached for a week.
export async function speciesNames() {
  try {
    const cached = JSON.parse(localStorage.getItem("fielddex.species") || "null");
    if (cached && Date.now() - cached.t < 7 * 864e5) return cached.names;
  } catch { /* ignore */ }
  const data = await getJSON(`${API}/pokemon-species?limit=2000`);
  const names = data.results.map((r) => title(r.name));
  try { localStorage.setItem("fielddex.species", JSON.stringify({ t: Date.now(), names })); } catch { /* ignore */ }
  return names;
}

// One dex entry, shaped like the entries in world.json (plus `image`).
export async function fetchEntry(query) {
  const q = String(query).trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  if (!q) throw new Error("not_found");
  const species = await getJSON(`${API}/pokemon-species/${encodeURIComponent(q)}`);
  const [mon, chain] = await Promise.all([
    getJSON(`${API}/pokemon/${species.id}`),
    species.evolution_chain?.url ? getJSON(species.evolution_chain.url).catch(() => null) : null,
  ]);
  const flavor = (species.flavor_text_entries || []).filter((f) => f.language?.name === "en");
  const name = english(species.names, "name") || title(species.name);
  return {
    number: species.id,
    name,
    key: species.name, // stable label for the recognizer
    category: english(species.genera, "genus"),
    types: mon.types.sort((a, b) => a.slot - b.slot).map((t) => t.type.name),
    height_m: mon.height / 10,
    weight_kg: mon.weight / 10,
    description: flavor.length ? clean(flavor[flavor.length - 1].flavor_text) : "",
    evolutions: chain ? chainNames(chain.chain) : [name],
    image: mon.sprites?.other?.["official-artwork"]?.front_default || mon.sprites?.front_default || null,
  };
}

// Offline stand-in for tests (?mock=...): a fixed entry for any name.
export function mockEntry(query) {
  const name = title(String(query).trim().toLowerCase().replace(/\s+/g, "-"));
  return {
    number: 1, name, key: name.toLowerCase(), category: "Test Pokémon", types: ["grass", "poison"],
    height_m: 0.7, weight_kg: 6.9, description: `${name} entry used for offline tests.`,
    evolutions: [name], image: null,
  };
}
