"""Download every split the experiment reads and record what arrived.

    python -m general_models_vs_plms.fetch_data --data DIR
"""

from __future__ import annotations

import argparse
import json

from datetime import UTC, datetime
from pathlib import Path
from tqdm import tqdm

from . import flip2, postcutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.data / "data_manifest.json"
    if manifest_path.exists():
        print(f"{manifest_path} exists; nothing to fetch")
        return

    files = {}
    for split_info in tqdm(flip2.SPLITS, desc="FLIP2 splits"):
        path = flip2.download_split(split_info, args.data)
        parts = flip2.load_split(path)["part"].value_counts()
        files[split_info.name] = {"url": split_info.url, "sha256": flip2.file_sha256(path),
                            "rows": {part: int(parts.get(part, 0)) for part in flip2.PARTS}}

    provenance = postcutoff.write_splits(args.data)
    for split_info in postcutoff.SPLITS:
        path = flip2.split_path(args.data, split_info)
        parts = flip2.load_split(path)["part"].value_counts()
        files[split_info.name] = {"source": postcutoff.SCORE_SET, "sha256": flip2.file_sha256(path),
                            "rows": {part: int(parts.get(part, 0)) for part in flip2.PARTS}}

    manifest = {"retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"), "files": files,
                "postcutoff": provenance}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path} for {len(files)} splits")


if __name__ == "__main__":
    main()
