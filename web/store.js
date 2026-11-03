// Tiny IndexedDB key-value store for what the phone learns (embeddings can run to megabytes,
// more than localStorage allows). Falls back to memory if IndexedDB is unavailable.

const memory = new Map();
let dbp = null;

function open() {
  if (!("indexedDB" in self)) return Promise.resolve(null);
  dbp ||= new Promise((resolve) => {
    const req = indexedDB.open("fielddex", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("kv");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
  return dbp;
}

export async function get(key) {
  const db = await open();
  if (!db) return memory.get(key);
  return new Promise((resolve) => {
    const req = db.transaction("kv").objectStore("kv").get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(undefined);
  });
}

export async function set(key, value) {
  const db = await open();
  if (!db) return void memory.set(key, value);
  return new Promise((resolve, reject) => {
    const tx = db.transaction("kv", "readwrite");
    tx.objectStore("kv").put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
