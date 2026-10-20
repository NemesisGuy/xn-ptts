// A Hugging Face model repo as a model source, named `hf://owner/name` or
// `hf://owner/name@revision` as in the native frontends and huggingface_hub. One request to
// the Hub API gives both the commit and the file list, and the files are picked by the same
// rules as the native Hub loader. A voice catalog on the Hub is named and pinned the same way.

import { openCache } from './fetch.js';

const HUB = 'https://huggingface.co';

/**
 * The model `PhononTTS.load` uses when none is given, pinned to a commit. A copy of
 * `DEFAULT_MODEL` in `ptts/src/checkpoint.rs`; the tests hold the two together.
 */
export const DEFAULT_MODEL = 'hf://davencyw/pocket-tts-ptts@7cc3c90b88f3e6617ded1b9b74100d1cd0162d90';

/**
 * The voice catalog `PhononTTS.load` and `catalogVoices` use when none is given, pinned to a
 * commit. A copy of `DEFAULT_VOICE_CATALOG` in `ptts/src/voice_catalog.rs`; the tests hold the
 * two together.
 */
export const DEFAULT_VOICE_CATALOG = 'hf://gradium/phonon-voices@da6c9ec0d47be8df5c87c730a3aa97187f3229c3';

/** `hf://owner/name` or `hf://owner/name@revision`. */
const HUB_SOURCE = /^hf:\/\/([\w.-]+\/[\w.-]+)(?:@(\S+))?$/;
/** The catalog file at the root of a voice catalog repo, as `CATALOG_FILE` in Rust. */
const CATALOG_FILE = 'voices.jsonl';

/** Weights per quant, in the order the native `weight_candidates` tries them. */
const WEIGHTS = {
  q8: ['model.q8.gguf', 'model.safetensors'],
  f32: ['model.safetensors', 'model.q8.gguf'],
};
/** As `VOICE_DIRS` in the native loader: a name in both keeps the one in `voices/`. */
const VOICE_DIRS = ['voices', 'embeddings'];
/** A checkpoint's own voice, registered as `default` unless the voice directories have one. */
const DEFAULT_VOICE_FILE = 'default-voice.safetensors';

/**
 * The repo and revision `source` names. Anything but `hf://owner/name[@revision]` is an error.
 *
 * @param {string} source
 * @returns {{ repo: string, revision: string }}
 */
export function parseHubModel(source) {
  const match = HUB_SOURCE.exec(source);
  if (!match) {
    throw new TypeError(
      `model must be a ModelSpec or a Hugging Face repo, 'hf://owner/name' or ` +
        `'hf://owner/name@revision'; got ${JSON.stringify(source)}. ` +
        'A browser cannot list a folder URL, so a model served elsewhere needs a ModelSpec.',
    );
  }
  return { repo: match[1], revision: match[2] ?? 'main' };
}

/**
 * The ModelSpec of a public Hugging Face repo, with every URL pinned to the commit the
 * revision points at, so a cached download is never served after the repo changes.
 *
 * With `cache`, the Hub's answer is kept beside the downloads, and used when the Hub cannot be
 * reached or fails (a 5xx or 429), so a model whose files are cached still loads offline. A
 * branch then means the commit it pointed at when it was last looked up.
 *
 * @param {string} source `hf://owner/name` or `hf://owner/name@revision`
 * @param {{ cache?: boolean }} [options]
 * @returns {Promise<import('./index.js').ModelSpec>}
 */
export async function resolveHubModel(source, { cache = true } = {}) {
  const { repo, info } = await lookUp(source, parseHubModel(source), cache);

  const files = new Set((info.siblings ?? []).map((s) => s.rfilename));
  const url = (file) =>
    `${HUB}/${repo}/resolve/${info.sha}/${file.split('/').map(encodeURIComponent).join('/')}`;

  const weights = {};
  for (const [quant, names] of Object.entries(WEIGHTS)) {
    const found = names.find((name) => files.has(name));
    if (found) weights[quant] = url(found);
  }
  const missing = ['config.json', 'tokenizer.json'].filter((name) => !files.has(name));
  if (!weights.q8) missing.push(`weights (${WEIGHTS.q8.join(' or ')})`);
  if (missing.length > 0) {
    throw new Error(`${source} at ${info.sha} has no ${missing.join(', ')}`);
  }

  const voiceFiles = new Map();
  for (const dir of VOICE_DIRS) {
    for (const file of files) {
      if (!file.startsWith(`${dir}/`) || !file.endsWith('.safetensors')) continue;
      const name = file.slice(dir.length + 1, -'.safetensors'.length);
      if (name && !name.includes('/') && !voiceFiles.has(name)) voiceFiles.set(name, file);
    }
  }
  if (!voiceFiles.has('default') && files.has(DEFAULT_VOICE_FILE)) {
    voiceFiles.set('default', DEFAULT_VOICE_FILE);
  }

  const names = [...voiceFiles.keys()].sort();
  // As the native `default_voice`: `default` if there is one, else the first by name.
  const defaultVoice = voiceFiles.has('default') ? 'default' : names[0];
  return {
    weights,
    tokenizer: url('tokenizer.json'),
    config: url('config.json'),
    voices: Object.fromEntries(names.map((name) => [name, url(voiceFiles.get(name))])),
    ...(defaultVoice ? { defaultVoice } : {}),
  };
}

/**
 * The repo and revision of a voice catalog on the Hub, or `null` when `source` does not start
 * with `hf://`. A malformed `hf://` name is an error.
 *
 * @param {string | URL} source
 * @returns {{ repo: string, revision: string } | null}
 */
export function parseHubCatalog(source) {
  if (!String(source).startsWith('hf://')) return null;
  const match = HUB_SOURCE.exec(source);
  if (!match) {
    throw new TypeError(
      "a voice catalog on the Hub is 'hf://owner/name' or 'hf://owner/name@revision'; " +
        `got ${JSON.stringify(String(source))}`,
    );
  }
  return { repo: match[1], revision: match[2] ?? 'main' };
}

/**
 * The URL of a Hub voice catalog's `voices.jsonl`, pinned to the commit the revision points
 * at, as `resolveHubModel` pins a model's files. The Hub's answer is cached the same way.
 *
 * @param {string} source `hf://owner/name` or `hf://owner/name@revision`
 * @param {{ cache?: boolean }} [options]
 * @returns {Promise<string>}
 */
export async function resolveHubCatalog(source, { cache = true } = {}) {
  const { repo, info } = await lookUp(source, parseHubCatalog(source), cache);
  if (!(info.siblings ?? []).some((s) => s.rfilename === CATALOG_FILE)) {
    throw new Error(`${source} at ${info.sha} has no ${CATALOG_FILE}`);
  }
  return `${HUB}/${repo}/resolve/${info.sha}/${CATALOG_FILE}`;
}

/** The Hub's answer for `repo` at `revision`: the commit it points at, and the files there. */
async function lookUp(source, { repo, revision }, cache) {
  const api = `${HUB}/api/models/${repo}/revision/${encodeURIComponent(revision)}`;
  const store = cache ? await openCache() : null;
  let response;
  try {
    response = await fetch(api);
  } catch (e) {
    response = await store?.match(api).catch(() => undefined);
    if (!response) {
      throw new Error(`cannot reach the Hugging Face Hub to look up ${source}: ${e.message}`);
    }
  }
  if (response.status >= 500 || response.status === 429) {
    response = (await store?.match(api).catch(() => undefined)) ?? response;
  }
  if (response.status === 404 && response.headers.get('x-error-code') === 'RevisionNotFound') {
    throw new Error(`${repo} has no revision '${revision}'`);
  }
  const info = response.ok
    ? await response.json().catch(() => {
        throw new Error(`cannot look up ${source}: the Hub's answer is not JSON`);
      })
    : null;
  if ([401, 403, 404].includes(response.status) || info?.gated) {
    throw new Error(
      `cannot load ${source}: the repo is missing, private or gated, and the browser can only ` +
        'load public repos',
    );
  }
  if (!response.ok) throw new Error(`cannot look up ${source}: HTTP ${response.status}`);
  await store?.put(api, new Response(JSON.stringify(info))).catch(() => {});
  return { repo, info };
}
