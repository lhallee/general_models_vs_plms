"""Download files of a run's private Hub dataset, as `publish` uploaded them, to a local folder.

    python -m general_models_vs_plms.pull --repository Synthyra/<name> --out DIR [--only PATTERN ...]

The Hub client reads its token from the HF_TOKEN environment variable, as `publish` does on the machine.
"""

from __future__ import annotations

import argparse

from pathlib import Path
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", help="a commit of the dataset; its newest by default")
    parser.add_argument("--only", nargs="*", help="download only the paths these patterns match, such as tables/**")
    args = parser.parse_args()

    folder = snapshot_download(args.repository, repo_type="dataset", revision=args.revision, local_dir=str(args.out),
                               allow_patterns=args.only or None)
    print(f"downloaded {args.repository} to {folder}")


if __name__ == "__main__":
    main()
