"""Download the official HERB 2.0 files used for external validation."""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "http://47.92.70.12/download/file/"
SERVER_ROOT = "/www/wwwroot/47.92.70.12/HERB_web/static/download_data/V2"
FILES = (
    "HERB_herb_info_v2.txt",
    "HERB_clinical_trials_v2.txt",
    "HERB_meta_info_v2.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        query = urllib.parse.urlencode({"file_path": f"{SERVER_ROOT}/{name}"})
        url = f"{BASE_URL}?{query}"
        destination = out_dir / name
        print(f"downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            destination.write_bytes(response.read())
        print(f"wrote {destination} ({destination.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
