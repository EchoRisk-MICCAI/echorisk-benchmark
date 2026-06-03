"""Command-line interface for scoring EchoRisk-MICCAI 2026 submissions.

Usage:
    echorisk-score --task 1 --predictions preds.csv --ground-truth gt.csv
    echorisk-score --task 2 --predictions preds.csv --ground-truth gt.csv
    echorisk-score --task 3 --predictions preds.csv --ground-truth gt.csv

Output is a JSON dictionary of metric values printed to stdout.
Warnings (e.g. extra predictions) are printed to stderr with a "warning:" prefix.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

from echorisk_eval.metrics import score_task1, score_task2, score_task3

TaskId = Literal[1, 2, 3]


def _score(task: TaskId, predictions_path: Path, ground_truth_path: Path) -> dict:
    """Dispatch to the appropriate task scoring function."""
    predictions = pd.read_csv(predictions_path)
    ground_truth = pd.read_csv(ground_truth_path)
    if task == 1:
        result = score_task1(predictions, ground_truth)
    elif task == 2:
        result = score_task2(predictions, ground_truth)
    elif task == 3:
        result = score_task3(predictions, ground_truth)
    else:
        raise ValueError(f"Unknown task: {task}. Expected 1, 2, or 3.")
    return dataclasses.asdict(result)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX exit code."""
    parser = argparse.ArgumentParser(
        description="Score a submission for the EchoRisk-MICCAI 2026 Challenge.",
    )
    parser.add_argument(
        "--task", type=int, choices=[1, 2, 3], required=True, help="Task number."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the predictions CSV.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to the ground-truth CSV.",
    )
    args = parser.parse_args(argv)

    # Route Python warnings to stderr with a consistent prefix so participants
    # see them even when piping stdout to a file.
    warnings.showwarning = lambda msg, *_: print(f"warning: {msg}", file=sys.stderr)

    try:
        result = _score(args.task, args.predictions, args.ground_truth)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Serialise NaN/Inf as null — bare NaN is not valid JSON per RFC 8259
    # and will be rejected by strict parsers (JS, jq, Synapse's harness).
    safe = {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in result.items()
    }
    print(json.dumps(safe, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())