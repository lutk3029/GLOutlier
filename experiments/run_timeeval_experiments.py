#!/usr/bin/env python3
"""Run GLOutlier on exactly the 48 TimeEval datasets used in the paper.

This is an orchestration script. It locates TimeEval's canonical *.test.csv and
optional *.train.csv files, then calls run_timeeval.py once per manifest row.
It never downloads or copies the upstream datasets into this repository.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_file(root, collection, dataset, kind):
    candidates = []
    wanted = normalized(dataset)
    for path in root.rglob(f"*.{kind}.csv"):
        if normalized(path.name.split(f".{kind}.csv")[0]) != wanted:
            continue
        score = 0 if normalized(collection) in normalized(str(path.parent)) else 1
        candidates.append((score, len(path.parts), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        raise RuntimeError(f"ambiguous {kind} file for {collection}/{dataset}: "
                           f"{candidates[0][2]} and {candidates[1][2]}")
    return candidates[0][2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="directory containing extracted TimeEval archives")
    parser.add_argument("--manifest", type=Path,
                        default=Path("experiments/timeeval_48.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("timeeval_runs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--only", action="append", default=[], metavar="COLLECTION/DATASET")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not args.manifest.is_absolute():
        args.manifest = repo_root / args.manifest

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if len(rows) != 48:
        raise RuntimeError(f"manifest must contain 48 datasets, found {len(rows)}")
    selected = set(args.only)
    completed = 0
    for row in rows:
        key = f"{row['collection']}/{row['dataset']}"
        if selected and key not in selected:
            continue
        test_path = find_file(args.dataset_root, row["collection"], row["dataset"], "test")
        if test_path is None:
            raise FileNotFoundError(f"missing TimeEval test file for {key}")
        train_path = find_file(args.dataset_root, row["collection"], row["dataset"], "train")
        command = [args.python, str(repo_root / "run_timeeval.py"), "--collection", row["collection"],
                   "--dataset", row["dataset"], "--test", str(test_path),
                   "--learning-type", row["learning_type"], "--output-dir",
                   str(args.output_dir / row["collection"] / row["dataset"]),
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--device", args.device]
        if train_path is not None:
            command += ["--train", str(train_path)]
        print("RUN", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
        completed += 1
    if selected and completed != len(selected):
        known = {f"{r['collection']}/{r['dataset']}" for r in rows}
        raise RuntimeError(f"unknown --only values: {sorted(selected - known)}")
    print(f"{'planned' if args.dry_run else 'completed'} {completed} dataset run(s)")


if __name__ == "__main__":
    main()
