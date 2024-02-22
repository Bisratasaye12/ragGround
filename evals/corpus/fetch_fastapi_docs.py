"""Fetch the eval corpus: FastAPI's English tutorial docs at a pinned release tag.

    python evals/corpus/fetch_fastapi_docs.py

FastAPI is MIT-licensed (https://github.com/fastapi/fastapi/blob/master/LICENSE). Files are
written to evals/corpus/fastapi/ (gitignored) with a manifest of SHA-256 hashes so eval runs
are reproducible and drift is detectable.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

TAG = "0.115.0"
PREFIX = "docs/en/docs/tutorial/"
OUT = Path(__file__).parent / "fastapi"
API = f"https://api.github.com/repos/fastapi/fastapi/git/trees/{TAG}?recursive=1"
RAW = f"https://raw.githubusercontent.com/fastapi/fastapi/{TAG}/"


def get(url: str, attempts: int = 4) -> bytes:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return bytes(resp.read())
        except OSError as exc:
            if i == attempts - 1:
                raise
            print(f"  retry {url} ({exc})", file=sys.stderr)
            time.sleep(2**i)
    raise AssertionError("unreachable")


def main() -> None:
    tree = json.loads(get(API))["tree"]
    paths = sorted(
        t["path"] for t in tree if t["path"].startswith(PREFIX) and t["path"].endswith(".md")
    )
    manifest = {}
    for path in paths:
        rel = path.removeprefix("docs/en/docs/")
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = get(RAW + path)
        dest.write_bytes(data)
        manifest[rel] = hashlib.sha256(data).hexdigest()
        print(f"  {rel} ({len(data)} bytes)")
    (OUT / "MANIFEST.json").write_text(
        json.dumps({"source": "fastapi/fastapi", "tag": TAG, "files": manifest}, indent=2)
    )
    print(f"Fetched {len(manifest)} files into {OUT}")


if __name__ == "__main__":
    main()
