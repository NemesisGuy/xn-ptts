# Changelog

Changes that need action when upgrading. The full notes for each release are on
[GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases).

## Unreleased

### One model source everywhere

Every frontend now names its model with one string: a local folder (or a `config.json` in one),
or a Hugging Face repo as `hf://owner/name` or `hf://owner/name@revision`. Only the `hf://`
prefix means the Hub.

| Where | Before | Now |
|---|---|---|
| `ptts` command | `--dir PATH`, or `--repo OWNER/MODEL --revision REV` | `--model PATH` or `--model hf://OWNER/MODEL@REV` |
| `export_coreml`, `create_voice` examples | `--dir`, `--repo`, `--revision`, `--config` | `--model` |
| Python | `TTS(config=..., revision=...)` | `TTS(model)`, the first argument |
| Python `ptts` command | `--model OWNER/MODEL --revision REV` | `--model hf://OWNER/MODEL@REV` |
| `ptts-openai-server` | `--config`, `PTTS_CONFIG`, `--revision`, `PTTS_REVISION` | `--model`, `PTTS_MODEL` |
| `ptts-ws-server` | `--config`, `--revision` | `--model` |
| Browser (`phonon-tts`) | a `ModelSpec` | a `ModelSpec`, or `'hf://OWNER/MODEL@REV'` for a public repo |

Breaking:

- The `ptts` command no longer has `--repo` or `--revision`. The Python `ptts` command no longer
  has `--revision`.
- A bare `OWNER/MODEL` is now a local path. Its not-found error suggests `hf://OWNER/MODEL`.
- The `export_coreml`, `create_voice` and `say` examples download only when built with
  `--features hub`.
- `ptts-ws-server` no longer has `--revision`.

Still accepted, with a deprecation warning:

- `PTTS_CONFIG` and `PTTS_REVISION` (and `--config`, `--revision`) on `ptts-openai-server`, where a
  bare repo ID keeps its old meaning. Setting one of them beside `PTTS_MODEL` is an error.
- `config=` and `revision=` in Python, with the same meaning as before.
- `--dir` on the `ptts` command and `--config` on `ptts-ws-server`, as other names for `--model`.

For Rust callers, the `ptts` crate has a new `hub` feature. With it, `Checkpoint::locate`
downloads `hf://` sources; without it, the library never downloads.
