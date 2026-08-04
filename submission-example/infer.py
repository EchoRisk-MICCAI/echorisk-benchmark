"""Reference inference script for EchoRisk-MICCAI 2026 submissions.

This is a WORKING but DUMMY implementation: it produces syntactically
valid, in-range predictions without doing real inference, so you can
verify your Docker plumbing (mounts, GPU visibility, network isolation,
output schema) before wiring up your actual model.

Replace `predict` with your real model call. Everything else, manifest
loading, the missing-view convention, the row-coverage guarantee, and
the output schema, should not need to change; it implements the contract
in docs/HARNESS_SPEC.md exactly.

TASK_ID is baked in at build time (see Dockerfile) via a build ARG that
becomes an ENV variable in the image. It is not passed as a run-time
argument or flag: the container is invoked with no arguments.
"""
from __future__ import annotations

import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

INPUT_DIR: Final[Path] = Path("/input")
OUTPUT_DIR: Final[Path] = Path("/output")
MANIFEST_PATH: Final[Path] = INPUT_DIR / "manifest.csv"
VIDEOS_DIR: Final[Path] = INPUT_DIR / "videos"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "predictions.csv"

REQUIRED_MANIFEST_COLS: Final[tuple[str, ...]] = (
    "patient_id",
    "timepoint",
    "video_a4c",
    "video_a2c",
)


@dataclass(frozen=True)
class TaskConfig:
    """Output column name, valid range, and fallback value for one task."""

    output_col: str
    value_min: float
    value_max: float
    fallback: float  # used when a case cannot be processed — see §4 of the spec


_TASK_CONFIGS: Final[dict[int, TaskConfig]] = {
    1: TaskConfig(output_col="lvef_pred", value_min=0.0, value_max=100.0, fallback=55.0),
    2: TaskConfig(output_col="dysfunction_pred", value_min=0.0, value_max=1.0, fallback=0.5),
    3: TaskConfig(output_col="cardiotoxicity_pred", value_min=0.0, value_max=1.0, fallback=0.5),
}


def _load_task_id() -> int:
    """Read TASK_ID baked in at build time. Fails loudly if unset or invalid."""
    raw = os.environ.get("TASK_ID")
    if raw is None:
        print("error: TASK_ID is not set. This must be baked in at build time "
              "via --build-arg TASK_ID=<1|2|3>.", file=sys.stderr)
        sys.exit(1)
    try:
        task_id = int(raw)
    except ValueError:
        print(f"error: TASK_ID must be an integer, got '{raw}'.", file=sys.stderr)
        sys.exit(1)
    if task_id not in _TASK_CONFIGS:
        print(f"error: TASK_ID must be 1, 2, or 3; got {task_id}.", file=sys.stderr)
        sys.exit(1)
    return task_id


def _load_manifest() -> list[dict[str, str]]:
    """Read and validate the case manifest. Fails loudly on schema errors."""
    if not MANIFEST_PATH.exists():
        print(f"error: manifest not found at {MANIFEST_PATH}.", file=sys.stderr)
        sys.exit(1)

    with MANIFEST_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("error: manifest.csv has no header row.", file=sys.stderr)
            sys.exit(1)
        missing_cols = set(REQUIRED_MANIFEST_COLS) - set(reader.fieldnames)
        if missing_cols:
            print(
                f"error: manifest is missing required columns: {sorted(missing_cols)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        rows = list(reader)

    if len(rows) == 0:
        print("error: manifest.csv contains no case rows.", file=sys.stderr)
        sys.exit(1)
    return rows


def _resolve_view(filename: str) -> Path | None:
    """Resolve a manifest video filename to a path.

    Returns None if the manifest cell is an empty string (missing view per
    the manifest convention, not a file-existence check).
    """
    if filename.strip() == "":
        return None
    path = VIDEOS_DIR / filename
    if not path.exists():
        # File referenced but not found on disk. This is a data integrity
        # problem distinct from "missing view"; treat as a per-case failure
        # so the row still gets written with a fallback value (see §4).
        raise FileNotFoundError(f"referenced video not found: {path}")
    return path


def predict(
    task_id: int,
    patient_id: str,
    timepoint: str,
    a4c_path: Path | None,
    a2c_path: Path | None,
) -> float:
    """Run inference on a single case.

    THIS IS A DUMMY IMPLEMENTATION. Replace this function's body with a
    call into your real model. Keep the signature identical to the input
    contract already established. Raising an exception here is safe: the
    caller in main() catches it and falls back to a defensive default, so
    the output row is still written (see §4 of the harness spec).

    Args:
        task_id: 1, 2, or 3, resolved once at start-up from the image's
            baked-in TASK_ID.
        patient_id: from the manifest row.
        timepoint: from the manifest row.
        a4c_path: resolved path to the apical 4-chamber DICOM, or None if
            that view is absent for this case.
        a2c_path: resolved path to the apical 2-chamber DICOM, or None if
            that view is absent for this case.

    Returns:
        A prediction value in the valid range for `task_id`.
    """
    # Dummy logic: return a plausible constant-ish value with small jitter,
    # regardless of which views are present. Delete this and call your
    # real model instead.
    config = _TASK_CONFIGS[task_id]
    midpoint = (config.value_min + config.value_max) / 2.0
    spread = (config.value_max - config.value_min) * 0.05
    return midpoint + random.uniform(-spread, spread)


def main() -> None:
    task_id = _load_task_id()
    config = _TASK_CONFIGS[task_id]
    manifest_rows = _load_manifest()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_fallback = 0

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "timepoint", config.output_col])
        writer.writeheader()

        for row in manifest_rows:
            patient_id = row["patient_id"]
            timepoint = row["timepoint"]

            try:
                a4c_path = _resolve_view(row["video_a4c"])
                a2c_path = _resolve_view(row["video_a2c"])
                value = predict(task_id, patient_id, timepoint, a4c_path, a2c_path)
                if not (config.value_min <= value <= config.value_max):
                    raise ValueError(f"prediction {value} out of range for task {task_id}")
            except Exception as exc:  # noqa: BLE001 — intentional catch-all at the row boundary
                print(
                    f"warning: case ({patient_id}, {timepoint}) failed "
                    f"({exc!r}); using fallback value {config.fallback}.",
                    file=sys.stderr,
                )
                value = config.fallback
                n_fallback += 1

            writer.writerow(
                {"patient_id": patient_id, "timepoint": timepoint, config.output_col: value}
            )

    n_total = len(manifest_rows)
    print(f"Wrote {n_total} predictions to {OUTPUT_PATH} "
          f"({n_fallback} used the fallback value).")


if __name__ == "__main__":
    main()