"""How well a model knows a protein sequence, measured by filling in residues that were masked out.

    python -m general_models_vs_plms.sequence_knowledge select --data DIR --predictions FILE --out FILE
    python -m general_models_vs_plms.sequence_knowledge prompts --set FILE --out DIR
    python -m general_models_vs_plms.sequence_knowledge esmc --set FILE --out DIR --model esmc_600m
    python -m general_models_vs_plms.sequence_knowledge score --set FILE --prompts DIR --out FILE

A fraction of a sequence's residues is replaced by a mask, and the model is asked for the residue at each masked
position. The share it gets right, at several fractions, is the model's knowledge of that sequence. For a masked
language model the same masks give the same share, beside its pseudo-log-likelihood: the mean log-probability of each
residue with that one position masked. A sequence with its residues shuffled is the floor, since nothing but
composition is left to know.

`select` draws the set: the parent of each dataset that has one, the Rhomax queries a language model answered, and
shuffled copies of some of both. `prompts` writes the files a language model reads, each sequence once a file at one
fraction, so no file shows a residue that another fraction of the same sequence hides in it. `esmc` scores a masked
language model on the same masks. `score` reads a language model's answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from . import flip2, postcutoff
from .encoding import AMINO_ACIDS, parent_text
from .plm_features import ChainScorer
from .tasks import load_layout


LEVELS = (0.15, 0.50, 0.85)
MASK = "_"
PIECE = 60
BATCHES = 2
ASKED_RHODOPSINS = 48
SHUFFLED_OF_EACH = 6
RHOMAX_TASK = "rhomax-by_wild_type__k256__s0"
SEED = 0
INSTRUCTIONS = (
    "Each line below is a piece of a protein sequence in one-letter amino acid codes. Some residues have been replaced "
    "by an underscore. Replace every underscore with the single residue you judge most likely, and change nothing "
    "else. A sequence is cut into pieces of at most 60 residues, and the pieces of one sequence share its label and "
    "are numbered in order. Some of the sequences are natural proteins and some are not.\n\n"
    "Reply with every line, in the same order and the same form: the label, a tab, and the completed piece. A "
    "completed piece has exactly the length of the piece you were given and holds no underscore. Write nothing else.")


@dataclass(frozen=True)
class Entry:
    """One sequence of the set: where it comes from, and whether a language model is asked about it."""

    name: str
    kind: str
    sequence: str
    asked: bool
    row: int | None = None


def stable_seed(text: str) -> int:
    """A seed that depends on the text alone."""
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)


def shuffled(sequence: str) -> str:
    """The sequence's residues in an order drawn from the sequence itself."""
    return "".join(np.random.default_rng(stable_seed(sequence)).permutation(list(sequence)))


def masked_positions(entry: Entry, level: float) -> np.ndarray:
    """The positions hidden at a level: the same for every model, and drawn among the 20 standard residues."""
    standard = np.array([index for index, residue in enumerate(entry.sequence) if residue in AMINO_ACIDS])  # (s,)
    count = max(1, round(level * len(standard)))
    generator = np.random.default_rng([stable_seed(entry.name), round(level * 100)])
    return np.sort(generator.choice(standard, size=count, replace=False))  # (m,)


def corrupted(sequence: str, positions: np.ndarray) -> str:
    """The sequence with a mask at each position."""
    # positions: (m,)
    residues = list(sequence)
    for position in positions:
        residues[position] = MASK
    return "".join(residues)


def read_set(path: Path) -> list[Entry]:
    return [Entry(**fields) for fields in json.loads(path.read_text(encoding="utf-8"))]


def select(splits_folder: Path, predictions_path: Path, model: str) -> list[Entry]:
    """Dataset parents, the Rhomax queries the model answered, and shuffled copies of some of both."""
    entries = []
    for split_info in (*flip2.SPLITS, *postcutoff.SPLITS):
        frame = flip2.load_split(flip2.split_path(splits_folder, split_info))
        layout = load_layout(split_info, frame)
        if layout.parent is not None and not any(entry.name == f"parent/{split_info.dataset}" for entry in entries):
            entries.append(Entry(f"parent/{split_info.dataset}", "parent", parent_text(layout), asked=True))
    parents = list(entries)

    predictions = pd.read_csv(predictions_path, usecols=["task_id", "method", "row", "answered"])
    answered = predictions[(predictions["task_id"] == RHOMAX_TASK) & (predictions["method"] == model) & predictions["answered"]]
    rows = np.sort(answered["row"].to_numpy())  # (q,)
    assert len(rows) == len(set(rows)) and len(rows) >= ASKED_RHODOPSINS, f"{len(rows)} answered Rhomax queries"
    rhomax = flip2.load_split(flip2.split_path(splits_folder, flip2.split_by_name("rhomax/by_wild_type")))
    asked_rows = set(np.random.default_rng(SEED).permutation(rows)[:ASKED_RHODOPSINS].tolist())
    rhodopsins = [Entry(f"rhodopsin/{row}", "rhodopsin", rhomax["sequence"].iloc[row], asked=row in asked_rows, row=int(row))
                  for row in rows]
    entries += rhodopsins

    asked_rhodopsins = [entry for entry in rhodopsins if entry.asked]
    for kind, sources in (("shuffled_parent", parents), ("shuffled_rhodopsin", asked_rhodopsins)):
        entries += [Entry(f"shuffled/{source.name}", kind, shuffled(source.sequence), asked=True)
                    for source in sources[:SHUFFLED_OF_EACH]]
    assert len({entry.name for entry in entries}) == len(entries)
    return entries


def pieces(text: str) -> list[str]:
    return [text[start:start + PIECE] for start in range(0, len(text), PIECE)]


def write_prompts(entries: list[Entry], out: Path) -> list[dict[str, object]]:
    """One prompt file a batch and rotation. Across rotations each sequence meets each level once.

    The index, kept apart from the prompt files, maps each neutral label to its sequence and level.
    """
    asked = sorted((entry for entry in entries if entry.asked), key=lambda entry: stable_seed(entry.name))
    index = []
    for batch in range(BATCHES):
        members = asked[batch::BATCHES]
        for rotation in range(len(LEVELS)):
            folder = f"p{len(index) + 1:03d}"
            order = np.random.default_rng([SEED, batch, rotation]).permutation(len(members))  # (n,)
            lines, labels = [], {}
            for label_number, member_index in enumerate(order, start=1):
                entry, label = members[member_index], f"S{label_number:02d}"
                level = LEVELS[(member_index + rotation) % len(LEVELS)]
                labels[label] = {"name": entry.name, "level": level}
                text = corrupted(entry.sequence, masked_positions(entry, level))
                lines += [f"{label}.{number:02d}\t{piece}" for number, piece in enumerate(pieces(text), start=1)]
            (out / folder).mkdir(parents=True)
            (out / folder / "prompt.txt").write_text(INSTRUCTIONS + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            index.append({"folder": folder, "batch": batch, "rotation": rotation, "labels": labels})
    return index


def parse_answer(text: str) -> dict[str, str]:
    """Each line's label and completed piece. A line that is not a label, a tab or spaces, and a piece is left out."""
    found = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0].startswith("S") and "." in parts[0]:
            found[parts[0]] = parts[1].upper()
    return found


def unigram_guess(sequence: str, positions: np.ndarray) -> str:
    """The most common residue among those left visible: what composition alone gives."""
    # positions: (m,)
    visible = Counter(residue for index, residue in enumerate(sequence) if index not in set(positions.tolist()))
    return visible.most_common(1)[0][0]


def score_entry(entry: Entry, level: float, answer: dict[str, str], label: str) -> dict[str, object]:
    """Masked positions right, with every masked position of a piece that came back at another length counted wrong."""
    positions = masked_positions(entry, level)  # (m,)
    expected = pieces(entry.sequence)
    correct, misaligned, altered = 0, 0, 0
    for number, piece in enumerate(expected, start=1):
        returned = answer.get(f"{label}.{number:02d}", "")
        if len(returned) != len(piece):
            misaligned += 1
            continue
        start = (number - 1) * PIECE
        inside = positions[(positions >= start) & (positions < start + len(piece))] - start  # (m_piece,)
        hidden = set(inside.tolist())
        correct += sum(returned[index] == piece[index] for index in hidden)
        altered += any(returned[index] != piece[index] for index in range(len(piece)) if index not in hidden)
    guess = unigram_guess(entry.sequence, positions)
    return {"name": entry.name, "kind": entry.kind, "length": len(entry.sequence), "level": level, "masked": len(positions),
            "correct": correct, "pieces": len(expected), "misaligned_pieces": misaligned, "altered_pieces": altered,
            "unigram_correct": sum(entry.sequence[position] == guess for position in positions)}


def score(entries: list[Entry], prompts: Path) -> pd.DataFrame:
    """One row a sequence and level, from every prompt folder that holds an answer."""
    by_name = {entry.name: entry for entry in entries}
    rows = []
    for item in json.loads((prompts / "index.json").read_text(encoding="utf-8")):
        answer_path = prompts / item["folder"] / "answer.txt"
        if not answer_path.exists():
            continue
        answer = parse_answer(answer_path.read_text(encoding="utf-8"))
        for label, meaning in item["labels"].items():
            rows.append({"prompt": item["folder"], **score_entry(by_name[meaning["name"]], meaning["level"], answer, label)})
    return pd.DataFrame(rows)


def score_masked_model(entries: list[Entry], model: str, device_name: str) -> pd.DataFrame:
    """A masked language model on the same masks, and its pseudo-log-likelihood with one position masked at a time."""
    scorer = ChainScorer(model, torch.device(device_name))
    rows = []
    for entry in entries:
        truth = torch.tensor([AMINO_ACIDS.index(residue) if residue in AMINO_ACIDS else -1 for residue in entry.sequence])  # (l,)
        shared = {"name": entry.name, "kind": entry.kind, "length": len(entry.sequence), "model": model}
        table = torch.from_numpy(scorer.masked_marginals(entry.sequence))  # (l, 20)
        standard = truth >= 0  # (l,)
        one_at_a_time = table[standard].gather(1, truth[standard][:, None]).squeeze(1)  # (s,)
        rows.append({**shared, "level": 0.0, "masked": int(standard.sum()),
                     "correct": int((table[standard].argmax(dim=1) == truth[standard]).sum()),
                     "mean_log_probability": float(one_at_a_time.mean())})
        for level in LEVELS:
            positions = torch.from_numpy(masked_positions(entry, level))  # (m,)
            input_ids, attention_mask = scorer.tokenize([entry.sequence])  # (1, l + 2), (1, l + 2)
            input_ids[0, positions + 1] = scorer.tokenizer.mask_token_id
            logits, _ = scorer.forward(input_ids, attention_mask)  # (1, l + 2, v)
            residues = torch.log_softmax(logits[0, positions + 1], dim=-1)[:, scorer.residue_ids].cpu()  # (m, 20)
            rows.append({**shared, "level": level, "masked": len(positions),
                         "correct": int((residues.argmax(dim=1) == truth[positions]).sum()),
                         "mean_log_probability": float(residues.gather(1, truth[positions][:, None]).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--data", type=Path, required=True)
    choose.add_argument("--predictions", type=Path, required=True)
    choose.add_argument("--model", default="claude-mythos-5-2@harness-low")
    choose.add_argument("--out", type=Path, required=True)
    write = commands.add_parser("prompts")
    write.add_argument("--set", type=Path, required=True)
    write.add_argument("--out", type=Path, required=True)
    masked = commands.add_parser("esmc")
    masked.add_argument("--set", type=Path, required=True)
    masked.add_argument("--out", type=Path, required=True)
    masked.add_argument("--model", required=True)
    masked.add_argument("--device", default="cuda")
    read = commands.add_parser("score")
    read.add_argument("--set", type=Path, required=True)
    read.add_argument("--prompts", type=Path, required=True)
    read.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        entries = select(args.data, args.predictions, args.model)
        args.out.write_text(json.dumps([asdict(entry) for entry in entries], indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{len(entries)} sequences, {sum(entry.asked for entry in entries)} asked of a language model: {dict(Counter(entry.kind for entry in entries))}")
    elif args.command == "prompts":
        index = write_prompts(read_set(args.set), args.out)
        (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{len(index)} prompt files in {args.out}")
    elif args.command == "esmc":
        args.out.mkdir(parents=True, exist_ok=True)
        table = score_masked_model(read_set(args.set), args.model, args.device)
        table.to_csv(args.out / f"knowledge_{args.model}.csv", index=False, lineterminator="\n")
        totals = table.groupby("level")[["correct", "masked"]].sum()
        print(f"{args.model}: {len(table)} rows, share right by level {(totals['correct'] / totals['masked']).round(3).to_dict()}")
    else:
        table = score(read_set(args.set), args.prompts)
        table.to_csv(args.out, index=False, lineterminator="\n")
        print(f"{len(table)} rows from {table['prompt'].nunique()} answered prompts written to {args.out}")


if __name__ == "__main__":
    main()
