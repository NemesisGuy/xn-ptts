# /// script
# requires-python = ">=3.9"
# dependencies = [
#    "requests",
#    "huggingface_hub",
# ]
# ///
"""Download a Gradium voice as a voice file for a Phonon checkpoint.

The voice can come from Gradium's catalog, or be one you designed or cloned in Gradium Studio
or with clone-voice.py. Each checkpoint needs its own copy of a voice, so pass the checkpoint
you will speak with as --model. Needs a Gradium API key in GRADIUM_API_KEY.

    uv run scripts/download-voice.py --list
    uv run scripts/download-voice.py VOICE_ID --model hf://OWNER/MODEL -o voice.safetensors
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

API = "https://api.gradium.ai/api"


def check(r: requests.Response, *args, **kwargs) -> None:
    """Exits with the API's own message when a request fails."""
    if not r.ok:
        try:
            detail = r.json()["detail"]
        except (ValueError, KeyError):
            detail = r.text
        sys.exit(f"Gradium API error {r.status_code}: {detail}")


def model_ext(model: str) -> str:
    """The checkpoint's model id, `sig@epoch`, from its config.json."""
    if model.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, _, revision = model.removeprefix("hf://").partition("@")
        path = Path(hf_hub_download(repo, "config.json", revision=revision or None))
    else:
        path = Path(model)
        if path.is_dir():
            path = path / "config.json"
    model_id = json.loads(path.read_text()).get("model_id")
    if not model_id:
        sys.exit(f"{model} has no model_id in its config.json: Gradium voices need a Phonon checkpoint")
    return f"{model_id['sig']}@{model_id['epoch']}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("voice_id", nargs="?", help="the voice's ID, as --list or Gradium Studio shows it")
    p.add_argument("--model", help="checkpoint folder, config.json, or hf://OWNER/MODEL[@REVISION]")
    p.add_argument("-o", "--out", type=Path, help="output file (default: VOICE_ID.safetensors)")
    p.add_argument("--list", action="store_true", help="list the catalog and your own voices")
    args = p.parse_args()
    if not args.list and not (args.voice_id and args.model):
        p.error("give a voice ID and --model, or --list")

    key = os.environ.get("GRADIUM_API_KEY")
    if not key:
        sys.exit("set GRADIUM_API_KEY to your Gradium API key")
    s = requests.Session()
    s.headers["x-api-key"] = key
    s.hooks["response"] = check

    if args.list:
        r = s.get(f"{API}/voices/", params={"include_catalog": "true", "limit": 0})
        for v in r.json():
            print(f"{v['uid']}  {v.get('language') or '--'}  {v['name']}: {v.get('description') or ''}")
        return

    ext = model_ext(args.model)
    r = s.get(f"{API}/voices/embeddings/{args.voice_id}/{ext}/blob", params={"as_safetensors": "1"})
    out = args.out or Path(f"{args.voice_id}.safetensors")
    out.write_bytes(r.content)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
