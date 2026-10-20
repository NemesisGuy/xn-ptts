// Voice catalogs, as in `ptts/src/voice_catalog.rs`: a `voices.jsonl` with one voice per line,
// `name` and `filename`, the voice file's path relative to the catalog, and optionally
// `voice_id`, `language` and `description`. It lives in a Hugging Face repo or at any URL, and
// is read only when a voice the model does not have is asked for.

import { fetchBytes } from './fetch.js';
import { DEFAULT_VOICE_CATALOG, parseHubCatalog, resolveHubCatalog } from './hub.js';

export class VoiceCatalog {
  #source;
  #hub;
  #cache;
  /** The catalog's URL and voices, or their pending read. */
  #read = null;

  /**
   * Nothing is fetched until the voices are needed.
   *
   * @param {string | URL} source `hf://owner/name[@revision]`, or the URL of a `voices.jsonl`
   * @param {{ cache?: boolean }} [options]
   */
  constructor(source, { cache = true } = {}) {
    if (typeof source !== 'string' && !(source instanceof URL)) {
      throw new TypeError(
        "a voice catalog is 'hf://owner/name', 'hf://owner/name@revision' or the URL of a " +
          `voices.jsonl; got ${JSON.stringify(source)}`,
      );
    }
    this.#hub = parseHubCatalog(source) !== null;
    // Resolved against the page, as a model's URLs are, so that voice files resolve against it.
    this.#source = this.#hub ? source : new URL(source, globalThis.location?.href).href;
    this.#cache = cache;
  }

  /** The catalog's voices, in file order. The first call fetches the catalog. */
  async voices() {
    return (await this.#load()).voices;
  }

  /**
   * The voice `query` names: its name in any case and with or without accents, so `kaua` finds
   * `Kauã`, or its exact `voiceId`.
   */
  async find(query) {
    const folded = foldName(query);
    return (await this.voices()).find((v) => foldName(v.name) === folded || v.voiceId === query);
  }

  /** The URL of `voice`'s file, which has to be a path inside the catalog. */
  async fileUrl({ name, filename }) {
    if (/[\\:]/.test(filename) || filename.split('/').some((p) => p === '' || p === '.' || p === '..')) {
      throw new Error(`voice '${name}' names the file '${filename}', which is not a path inside the catalog`);
    }
    const { url } = await this.#load();
    return new URL(filename.split('/').map(encodeURIComponent).join('/'), url).href;
  }

  #load() {
    this.#read ??= this.#fetch().catch((e) => {
      this.#read = null; // let a later call retry
      throw e;
    });
    return this.#read;
  }

  async #fetch() {
    const url = this.#hub ? await resolveHubCatalog(this.#source, { cache: this.#cache }) : this.#source;
    const text = new TextDecoder().decode(await fetchBytes(url, { cache: this.#cache }));
    const voices = [];
    for (const [i, line] of text.split('\n').entries()) {
      if (line.trim() === '') continue;
      let voice;
      try {
        voice = JSON.parse(line);
      } catch (e) {
        throw new Error(`${url}:${i + 1}: ${e.message}`);
      }
      if (typeof voice?.name !== 'string' || typeof voice.filename !== 'string') {
        throw new Error(`${url}:${i + 1}: a voice needs a name and a filename`);
      }
      const { name, filename, voice_id: voiceId = null, language = null, description = null } = voice;
      voices.push({ name, filename, voiceId, language, description });
    }
    return { url, voices };
  }
}

/**
 * The voices of a catalog, without loading a model: only the catalog is fetched.
 *
 * @param {string | URL} [catalog] as `voiceCatalog` in `PhononTTS.load`
 * @param {{ cache?: boolean }} [options]
 * @returns {Promise<import('./index.js').CatalogVoice[]>}
 */
export async function catalogVoices(catalog = DEFAULT_VOICE_CATALOG, { cache = true } = {}) {
  const voices = await new VoiceCatalog(catalog, { cache }).voices();
  return voices.map(({ name, voiceId, language, description }) => ({ name, voiceId, language, description }));
}

/** `name` as voice lookups compare it, as the native `fold`: lower case, accents dropped. */
export function foldName(name) {
  return String(name)
    .toLowerCase()
    .replace(/\p{Script=Latin}/gu, (c) => (c === 'ø' ? 'o' : c.normalize('NFD')[0]));
}
