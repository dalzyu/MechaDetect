import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import os
import concurrent.futures
from huggingface_hub import get_token

token = get_token()
if token:
    os.environ["HF_TOKEN"] = token

from datasets import load_dataset
from scripts.data_prep.acquire_all_images import HF_SOURCES

def test_source(name, cfg):
    repo = cfg["repo"]
    split = cfg.get("split", "train")
    try:
        ds = load_dataset(repo, split=split, streaming=True, token=token)
        first = next(iter(ds))
        keys = list(first.keys())
        img_key = None
        for k in keys:
            v = first[k]
            if hasattr(v, "size") or (isinstance(v, dict) and ("bytes" in v or "path" in v)) or isinstance(v, bytes):
                img_key = k
                break
        return f"OK:   {name:32s} -> {repo:45s} img_key: {img_key}"
    except Exception as e:
        return f"FAIL: {name:32s} -> {repo:45s} err: {type(e).__name__}: {str(e)[:70]}"

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(test_source, k, v): k for k, v in HF_SOURCES.items()}
    for fut in concurrent.futures.as_completed(futs):
        print(fut.result())
