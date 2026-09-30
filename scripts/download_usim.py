#!/usr/bin/env python3
"""Download the USIM dataset from Hugging Face (via hf-mirror)."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO = "Vincent2025hello/usim"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def list_files(endpoint: str) -> list[str]:
    api = HfApi(endpoint=endpoint)
    info = api.dataset_info(REPO)
    return [s.rfilename for s in info.siblings]


def want(path: str, mode: str) -> bool:
    if path in {".gitattributes", "README.md"}:
        return True
    if mode == "all":
        return True
    if mode == "meta":
        return "/meta/" in path or path.endswith(".md")
    if mode == "parquet":
        return path.endswith(".parquet") or "/meta/" in path or path.endswith(".md")
    if mode == "videos":
        return path.endswith(".mp4")
    raise ValueError(mode)


def download_one(filename: str, dest: Path, endpoint: str) -> str:
    return hf_hub_download(
        repo_id=REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest),
        endpoint=endpoint,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/hy-tmp/data/usim")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--mode", choices=["meta", "parquet", "videos", "all"], default="parquet")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Listing files from {args.endpoint} ...", flush=True)
    files = [f for f in list_files(args.endpoint) if want(f, args.mode)]
    print(f"{len(files)} files to fetch (mode={args.mode})", flush=True)

    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, f, dest, args.endpoint): f for f in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="usim"):
            name = futs[fut]
            try:
                fut.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail.append((name, str(e)))
    print(json.dumps({"ok": ok, "fail": len(fail)}, indent=2))
    if fail:
        print("first failures:", fail[:8])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
