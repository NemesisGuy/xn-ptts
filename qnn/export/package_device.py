"""Compile the source models package.py uploaded for another device, and link them.

    PHONON_OUT=out/s24 python package_device.py "Samsung Galaxy S24 (Family)" out/s25

Reuses the source models and options of the compile jobs in <from>/package.json, so nothing
is exported or uploaded again. Writes the context binary and job ids into PHONON_OUT, in the
layout package.py writes, so bundle.py and device_check.py read either.
"""

import json
import os
import sys
from pathlib import Path

import qai_hub as hub

OUT = Path(os.environ.get("PHONON_OUT", Path(__file__).parent / "out"))
NAME = "phonon"


def main():
    device_name, source = sys.argv[1], Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    device = hub.Device(device_name)
    jobs = json.loads((source / "package.json").read_text())
    compiled = {}
    for name, jid in jobs["compile"].items():
        src = hub.get_job(jid)
        compiled[name] = hub.submit_compile_job(model=src.model, device=device, name=f"{NAME} {name}", options=src.options)
        print(name, "compile", compiled[name].job_id, compiled[name].url, flush=True)
    for name, job in compiled.items():
        status = job.wait()
        print(name, "compile", status.code, status.message or "", flush=True)
        if not status.success:
            sys.exit(1)
    link = hub.submit_link_job([j.get_target_model() for j in compiled.values()], device=device, name=NAME)
    print("link", link.job_id, link.url, flush=True)
    status = link.wait()
    print("link", status.code, status.message or "", flush=True)
    if not status.success:
        sys.exit(1)
    model = link.get_target_model()
    binary = OUT / f"{NAME}.bin"
    model.download(str(binary))
    record = {"device": device_name, "compile": {n: j.job_id for n, j in compiled.items()}, "link": link.job_id,
              "model": model.model_id, "binary_bytes": binary.stat().st_size}
    (OUT / "package.json").write_text(json.dumps(record, indent=2))
    print("context binary", binary, binary.stat().st_size / 2**20, "MiB", flush=True)


if __name__ == "__main__":
    main()
