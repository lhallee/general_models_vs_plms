"""Upload a run's state folder to a private Hub dataset, leaving out the Hub cache and the smoke run.

    python -m general_models_vs_plms.publish --state DIR --repository Synthyra/<name> [--only PATTERN ...]
"""

from __future__ import annotations

import argparse

from pathlib import Path
from huggingface_hub import HfApi


LEFT_OUT = ["hf/**", "features_smoke/**", "bundle.tar.gz"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--only", nargs="*", help="upload only the paths these patterns match, such as tables/**")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repository, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(folder_path=str(args.state), repo_id=args.repository, repo_type="dataset",
                               allow_patterns=args.only or None, ignore_patterns=LEFT_OUT,
                               commit_message="Run state of the FLIP2 in-context experiment")
    print(f"uploaded {args.state} to {args.repository} at {commit.oid}")


if __name__ == "__main__":
    main()
