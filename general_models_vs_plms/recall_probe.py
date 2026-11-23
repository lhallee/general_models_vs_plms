"""Does a model know the benchmark's parent sequences? A continuation probe with a shuffled control.

    python -m general_models_vs_plms.recall_probe run --data DIR --out DIR --model claude-fable-5-2 --effort low --budget 150
    python -m general_models_vs_plms.recall_probe export --data DIR --out DIR --index FILE
    python -m general_models_vs_plms.recall_probe summarize --data DIR --probes DIR --out FILE

The model reads the start of a parent sequence and writes the next residues. The control shuffles the parent's
residues first, which keeps its composition and removes its order. A model that continues the real sequence and not
the shuffled one has that sequence, or close relatives of it, in memory, so withholding the protein's name from a
prompt does not hide which protein it is. `run` asks a model through the API, `export` writes the same questions as
prompt files for a model reached through subagents (`subagent_arm` audits and ingests them), and `summarize` scores
every model's records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import anthropic
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from pathlib import Path

from . import flip2, plan
from .claude_calls import PRICES, Prompt, SpendLedger, run_prompt
from .encoding import AMINO_ACIDS, CHAIN_BREAK, parent_text
from .tasks import load_layout, name_code


PROBE_SPLITS = ("amylase/by_mutation", "ired/two_to_many", "nucb/two_to_many", "trpb/one_to_many",
                "pdz3/single_to_double", "fgfr2/random")
CUT_FRACTIONS = (0.25, 0.5, 0.75)
CONTINUATION = 30


@dataclass(frozen=True)
class Probe:
    """One continuation question and its answer."""

    probe_id: str
    dataset: str
    kind: str
    cut: int
    shown: str
    expected: str


def build_probes(data_folder: Path) -> list[Probe]:
    """Three cuts of each parent chain, real and shuffled."""
    probes = []
    for split_info in plan.split_infos(PROBE_SPLITS):
        frame = flip2.load_split(flip2.split_path(data_folder, split_info))
        chain = parent_text(load_layout(split_info, frame)).split(CHAIN_BREAK)[0]
        generator = np.random.default_rng([name_code(split_info.dataset), 7])
        shuffled = "".join(generator.permutation(list(chain)))
        for kind, sequence in (("real", chain), ("shuffled", shuffled)):
            for fraction in CUT_FRACTIONS:
                cut = int(len(sequence) * fraction)
                probes.append(Probe(f"{split_info.dataset}__{kind}__{cut}", split_info.dataset, kind, cut, sequence[:cut],
                                    sequence[cut:cut + CONTINUATION]))
    return probes


def probe_prompt(probe: Probe) -> Prompt:
    """The question as a request; the whole text is the cacheable part and the suffix is empty of content."""
    text = (f"Here is the beginning of a protein sequence:\n\n{probe.shown}\n\nWrite the next {len(probe.expected)} "
            f"residues of this sequence.")
    return Prompt(probe.probe_id, text, f" Reply with the {len(probe.expected)} one-letter codes only.")


def continuation_identity(reply: str, expected: str) -> tuple[float, bool]:
    """Share of positions a reply gets right, and whether its first residue is right."""
    residues = [letter for letter in reply.upper() if letter in AMINO_ACIDS][:len(expected)]
    matches = sum(left == right for left, right in zip(residues, expected, strict=False))
    return matches / len(expected), bool(residues) and residues[0] == expected[0]


def export(data_folder: Path, out: Path, index_path: Path) -> list[dict[str, object]]:
    """Write each probe as a numbered prompt file for a model reached through subagents, with the index `subagent_arm` reads."""
    entries = []
    for probe in build_probes(data_folder):
        prompt = probe_prompt(probe)
        text = prompt.prefix + prompt.suffix
        folder = out / f"p{len(entries) + 1:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "prompt.txt").write_text(text, encoding="utf-8", newline="\n")
        entries.append({"folder": folder.name, "task_id": probe.probe_id, "chunk": 0, "record_id": probe.probe_id,
                        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"prompts": str(out), "entries": entries}, indent=1), encoding="utf-8")
    return entries


def summarize(data_folder: Path, probes_folder: Path) -> pd.DataFrame:
    """One row per model and probe that has a record: the share of the continuation it got right."""
    rows = []
    for folder in sorted(path for path in probes_folder.iterdir() if path.is_dir()):
        for probe in build_probes(data_folder):
            path = folder / f"{probe.probe_id}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            identity, first_right = continuation_identity(record["text"], probe.expected)
            rows.append({"method": folder.name.replace("__", "@"), "dataset": probe.dataset, "kind": probe.kind, "cut": probe.cut,
                         "identity": identity, "first_right": first_right, "stop_reason": record["stop_reason"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="ask a model through the API")
    runner.add_argument("--data", type=Path, required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--model", choices=sorted(PRICES), required=True)
    runner.add_argument("--effort", required=True)
    runner.add_argument("--budget", type=float, required=True)
    exporter = commands.add_parser("export", help="write the probes as prompt files for a subagent arm")
    exporter.add_argument("--data", type=Path, required=True)
    exporter.add_argument("--out", type=Path, required=True)
    exporter.add_argument("--index", type=Path, required=True)
    reader = commands.add_parser("summarize", help="score every model's records")
    reader.add_argument("--data", type=Path, required=True)
    reader.add_argument("--probes", type=Path, required=True)
    reader.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        entries = export(args.data, args.out, args.index)
        print(f"wrote {len(entries)} probes to {args.out} and their index to {args.index}")
        return
    if args.command == "summarize":
        table = summarize(args.data, args.probes)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(table.groupby(["method", "kind"])["identity"].mean().unstack().round(3).to_string())
        return

    probes = build_probes(args.data)
    folder = args.out / "probes" / f"{args.model}__{args.effort}"
    folder.mkdir(parents=True, exist_ok=True)
    (args.out / "probes" / "probes.json").write_text(json.dumps([asdict(probe) for probe in probes], indent=1), encoding="utf-8")
    ledger = SpendLedger(args.out, f"probe__{args.model}__{args.effort}", args.budget)
    client = anthropic.Anthropic(max_retries=8, timeout=600.0)
    outcomes = [run_prompt(client, args.model, args.effort, probe_prompt(probe), folder, ledger, 2000) for probe in probes]
    print(f"probes {args.model} {args.effort}: { {outcome: outcomes.count(outcome) for outcome in set(outcomes)} }")


if __name__ == "__main__":
    main()
