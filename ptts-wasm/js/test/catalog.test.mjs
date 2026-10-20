// Voice catalogs, against a fake Hub and a fake network, so it runs under plain `node --test`
// with no network and no model.

import { test } from 'node:test';
import assert from 'node:assert/strict';

const { VoiceCatalog, catalogVoices, foldName } = await import('../catalog.js');
const { DEFAULT_VOICE_CATALOG } = await import('../hub.js');

const SHA = '0123456789abcdef0123456789abcdef01234567';
const BASE = `https://huggingface.co/acme/voices/resolve/${SHA}`;

const VOICES = [
  { filename: 'voices/a1.safetensors', name: 'Zoey', voice_id: 'a1', language: 'en', locale: 'en-us' },
  { filename: 'voices/b2.safetensors', name: 'Kauã', voice_id: 'b2', language: 'pt', description: 'Calm' },
  { filename: 'voices/c3 x.safetensors', name: 'Lucía-Sol' },
];

/**
 * A network with a Hub that puts every repo at `sha` with a `voices.jsonl` holding `text`, and
 * the same file at any other URL ending in `voices.jsonl`. Records what was asked.
 */
function fakeNetwork(text, { sha = SHA, files = ['voices.jsonl'] } = {}) {
  const requests = [];
  globalThis.fetch = async (url) => {
    url = String(url);
    requests.push(url);
    if (url.includes('/api/models/')) return Response.json({ sha, siblings: files.map((rfilename) => ({ rfilename })) });
    if (url.endsWith('/voices.jsonl')) return new Response(text);
    return new Response('', { status: 404 });
  };
  return requests;
}

const jsonl = (voices) => voices.map((v) => JSON.stringify(v)).join('\n') + '\n';

test('catalogVoices lists the default catalog from its voices.jsonl alone', async () => {
  const [repo, revision] = DEFAULT_VOICE_CATALOG.slice('hf://'.length).split('@');
  const requests = fakeNetwork(jsonl(VOICES), { sha: revision });
  assert.deepEqual(await catalogVoices(), [
    { name: 'Zoey', voiceId: 'a1', language: 'en', description: null },
    { name: 'Kauã', voiceId: 'b2', language: 'pt', description: 'Calm' },
    { name: 'Lucía-Sol', voiceId: null, language: null, description: null },
  ]);
  assert.deepEqual(requests, [
    `https://huggingface.co/api/models/${repo}/revision/${revision}`,
    `https://huggingface.co/${repo}/resolve/${revision}/voices.jsonl`,
  ]);
});

test('an unpinned catalog is pinned to the commit its revision points at', async () => {
  const requests = fakeNetwork(jsonl(VOICES));
  const catalog = new VoiceCatalog('hf://acme/voices');
  assert.equal(requests.length, 0, 'nothing is fetched until a voice is needed');
  const zoey = await catalog.find('zoey');
  assert.deepEqual(requests, [
    'https://huggingface.co/api/models/acme/voices/revision/main',
    `${BASE}/voices.jsonl`,
  ]);
  assert.equal(await catalog.fileUrl(zoey), `${BASE}/voices/a1.safetensors`);
  assert.equal(await catalog.fileUrl(await catalog.find('lucia-sol')), `${BASE}/voices/c3%20x.safetensors`);
  assert.equal(requests.length, 2, 'the catalog is read once');
});

test('names match in any case and with or without accents, as the native fold does', async () => {
  fakeNetwork(jsonl(VOICES));
  const catalog = new VoiceCatalog('hf://acme/voices');
  const name = async (query) => (await catalog.find(query))?.name;
  assert.equal(await name('Kauã'), 'Kauã');
  assert.equal(await name('kaua'), 'Kauã');
  assert.equal(await name('KAUA'), 'Kauã');
  assert.equal(await name('lucia-sol'), 'Lucía-Sol');
  assert.equal(await name('zoey'), 'Zoey');
  assert.equal(await name('nobody'), undefined);
  assert.equal(foldName('Bjørn'), 'bjorn');
  assert.equal(foldName('Straße'), 'straße');
});

test('voice ids match exactly', async () => {
  fakeNetwork(jsonl(VOICES));
  const catalog = new VoiceCatalog('hf://acme/voices');
  assert.equal((await catalog.find('b2'))?.name, 'Kauã');
  assert.equal(await catalog.find('B2'), undefined);
});

test('a catalog at a URL has its voice files beside it', async () => {
  const requests = fakeNetwork(jsonl(VOICES));
  const catalog = new VoiceCatalog(new URL('https://example.com/v/voices.jsonl'));
  const zoey = await catalog.find('Zoey');
  assert.equal(await catalog.fileUrl(zoey), 'https://example.com/v/voices/a1.safetensors');
  assert.deepEqual(requests, ['https://example.com/v/voices.jsonl']);
});

test('a file outside the catalog is refused', async () => {
  const bad = ['../secret.safetensors', 'voices/../../x', '/etc/x', './x', 'a//b', '', 'C:/x', 'a\\b', 'https://evil.example/x'];
  fakeNetwork(jsonl(bad.map((filename, i) => ({ name: `v${i}`, filename }))));
  const catalog = new VoiceCatalog('hf://acme/voices');
  for (const voice of await catalog.voices()) {
    await assert.rejects(catalog.fileUrl(voice), /names the file .*, which is not a path inside the catalog/, voice.filename);
  }
});

test('a bad line is reported with its number', async () => {
  fakeNetwork('{"name": "A", "filename": "a.safetensors"}\n\nnot json\n');
  await assert.rejects(catalogVoices('hf://acme/voices'), /voices\.jsonl:3: /);
  fakeNetwork('{"name": "A"}\n');
  await assert.rejects(catalogVoices('hf://acme/voices'), /voices\.jsonl:1: a voice needs a name and a filename/);
});

test('a repo without a voices.jsonl says so', async () => {
  fakeNetwork('', { files: ['README.md'] });
  await assert.rejects(catalogVoices('hf://acme/voices'), new RegExp(`hf://acme/voices at ${SHA} has no voices\\.jsonl`));
});

test('a private or missing catalog repo says the browser reads public repos only', async () => {
  globalThis.fetch = async () => new Response('{}', { status: 401 });
  await assert.rejects(catalogVoices('hf://acme/voices'), /cannot load hf:\/\/acme\/voices: .*public repos/);
});

test('a malformed catalog is refused before any request', async () => {
  const requests = fakeNetwork('');
  assert.throws(() => new VoiceCatalog('hf://acme'), /voice catalog on the Hub is 'hf:\/\/owner\/name'/);
  for (const bad of [null, 5, {}]) {
    assert.throws(() => new VoiceCatalog(bad), /a voice catalog is .* or the URL of a voices\.jsonl/, String(bad));
  }
  assert.equal(requests.length, 0);
});

test('a catalog that failed to read is read again next time', async () => {
  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch');
  };
  const catalog = new VoiceCatalog('https://example.com/voices.jsonl', { cache: false });
  await assert.rejects(catalog.find('zoey'), /Failed to fetch/);
  fakeNetwork(jsonl(VOICES));
  assert.equal((await catalog.find('zoey'))?.name, 'Zoey');
});
