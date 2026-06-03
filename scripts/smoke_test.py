"""End-to-end smoke tests for EchoRisk-MICCAI 2026 scoring code.

Simulates a full participant submission cycle at challenge-realistic scale:
generates synthetic ground truth (with realistic multi-timepoint structure
for Tasks 1 and 2, baseline-only T1 for Task 3), generates several categories
of submissions, scores each through the CLI, and verifies the expected
behaviour.

Run from the repository root with the venv active:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Challenge-realistic cohort sizes.
# Task 1 & 2: 339 patients averaging ~2.4 exams each -> 814 exams. We generate
# a smaller validation-scale sample here: ~60 patients across T1-T3.
# Task 3: baseline only, 80 patients (smaller cohort).
N_PATIENTS_LONGITUDINAL = 60
N_TASK3 = 80
TIMEPOINTS = ["T1", "T2", "T3"]  # subset for smoke test realism

RNG_SEED = 20260415


@dataclass(frozen=True)
class SmokeCase:
    """A single smoke-test case describing one submission scenario."""

    name: str
    task: int
    predictions_builder: Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]
    expect_success: bool
    expected_error_fragment: str | None = None


# ---------------------------------------------------------------------------
# Ground-truth generation
# ---------------------------------------------------------------------------


def make_task1_gt(rng: np.random.Generator) -> pd.DataFrame:
    """Task 1 ground truth: longitudinal LVEF at multiple timepoints.

    Each patient contributes 2-3 exams across T1-T3. LVEF distribution
    approximates a cardio-oncology cohort followed on-treatment.
    """
    rows = []
    for i in range(1, N_PATIENTS_LONGITUDINAL + 1):
        pid = f"ECHORISK_{i:04d}"
        n_exams = rng.integers(2, 4)  # 2 or 3 exams per patient
        timepoints = rng.choice(TIMEPOINTS, size=n_exams, replace=False)
        for tp in sorted(timepoints):
            lvef = float(np.clip(rng.normal(58.0, 7.0), 20.0, 80.0))
            rows.append(
                {
                    "patient_id": pid,
                    "timepoint": str(tp),
                    "video_a4c": f"{pid}_{tp}_A4C.dcm",
                    "video_a2c": f"{pid}_{tp}_A2C.dcm",
                    "lvef": round(lvef, 1),
                    "biomarker_elevated": int(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


def make_task2_gt(rng: np.random.Generator) -> pd.DataFrame:
    """Task 2 ground truth: binary LV dysfunction across timepoints, ~17% dysfunction rate."""
    rows = []
    for i in range(1, N_PATIENTS_LONGITUDINAL + 1):
        pid = f"ECHORISK_{i:04d}"
        n_exams = rng.integers(2, 4)
        timepoints = rng.choice(TIMEPOINTS, size=n_exams, replace=False)
        for tp in sorted(timepoints):
            rows.append(
                {
                    "patient_id": pid,
                    "timepoint": str(tp),
                    "video_a4c": f"{pid}_{tp}_A4C.dcm",
                    "video_a2c": f"{pid}_{tp}_A2C.dcm",
                    "lv_dysfunction": int(rng.binomial(1, 0.17)),
                }
            )
    return pd.DataFrame(rows)


def make_task3_gt(rng: np.random.Generator) -> pd.DataFrame:
    """Task 3 ground truth: baseline-only (T1) with ~65% cardiotoxicity prevalence."""
    rows = []
    for i in range(1, N_TASK3 + 1):
        pid = f"ECHORISK_{i:04d}"
        rows.append(
            {
                "patient_id": pid,
                "timepoint": "T1",
                "video_a4c": f"{pid}_T1_A4C.dcm",
                "video_a2c": f"{pid}_T1_A2C.dcm",
                "cardiotoxicity": int(rng.binomial(1, 0.65)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Submission builders
# ---------------------------------------------------------------------------


def task1_perfect(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "lvef_pred": gt["lvef"].values,
        }
    )


def task1_realistic(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    noise = rng.normal(0.0, 3.5, size=len(gt))
    preds = np.clip(gt["lvef"].values + noise, 0.0, 100.0)
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "lvef_pred": preds.round(2),
        }
    )


def task1_constant(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "lvef_pred": np.full(len(gt), 58.0),
        }
    )


def task1_wrong_column(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "prediction": gt["lvef"].values,
        }
    )


def task1_missing_timepoint(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Submission missing the timepoint column."""
    return pd.DataFrame(
        {"patient_id": gt["patient_id"].values, "lvef_pred": gt["lvef"].values}
    )


def task1_out_of_range(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    preds = task1_realistic(gt, rng)
    preds.loc[0, "lvef_pred"] = 150.0
    return preds


def task1_missing_exams(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    preds = task1_realistic(gt, rng)
    return preds.iloc[:-5].copy()


def task2_perfect(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    probs = np.where(gt["lv_dysfunction"].values == 1, 0.99, 0.01)
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "dysfunction_pred": probs,
        }
    )


def task2_realistic(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    y = gt["lv_dysfunction"].values
    probs = np.where(
        y == 1,
        rng.beta(5.0, 2.5, size=len(gt)),
        rng.beta(2.5, 5.0, size=len(gt)),
    )
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "dysfunction_pred": probs.round(4),
        }
    )


def task2_constant_half(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "dysfunction_pred": np.full(len(gt), 0.5),
        }
    )


def task2_out_of_range(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    preds = task2_realistic(gt, rng)
    preds.loc[0, "dysfunction_pred"] = 1.5
    return preds


def task3_perfect(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    probs = np.where(gt["cardiotoxicity"].values == 1, 0.95, 0.05)
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "cardiotoxicity_pred": probs,
        }
    )


def task3_realistic(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    y = gt["cardiotoxicity"].values
    probs = np.where(
        y == 1,
        rng.beta(4.0, 2.5, size=len(gt)),
        rng.beta(2.0, 5.0, size=len(gt)),
    )
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "cardiotoxicity_pred": probs.round(4),
        }
    )


def task3_random(gt: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": gt["patient_id"].values,
            "timepoint": gt["timepoint"].values,
            "cardiotoxicity_pred": rng.uniform(0.0, 1.0, size=len(gt)),
        }
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


CASES: list[SmokeCase] = [
    # Task 1 — happy paths
    SmokeCase("task1_perfect", 1, task1_perfect, expect_success=True),
    SmokeCase("task1_realistic", 1, task1_realistic, expect_success=True),
    SmokeCase("task1_constant", 1, task1_constant, expect_success=True),
    # Task 1 — rejections
    SmokeCase(
        "task1_wrong_column", 1, task1_wrong_column,
        expect_success=False, expected_error_fragment="missing required columns",
    ),
    SmokeCase(
        "task1_missing_timepoint", 1, task1_missing_timepoint,
        expect_success=False, expected_error_fragment="missing required columns",
    ),
    SmokeCase(
        "task1_out_of_range", 1, task1_out_of_range,
        expect_success=False, expected_error_fragment="LVEF range",
    ),
    SmokeCase(
        "task1_missing_exams", 1, task1_missing_exams,
        expect_success=False, expected_error_fragment="must have a prediction",
    ),
    # Task 2 — happy paths
    SmokeCase("task2_perfect", 2, task2_perfect, expect_success=True),
    SmokeCase("task2_realistic", 2, task2_realistic, expect_success=True),
    SmokeCase("task2_constant_half", 2, task2_constant_half, expect_success=True),
    # Task 2 — rejections
    SmokeCase(
        "task2_out_of_range", 2, task2_out_of_range,
        expect_success=False, expected_error_fragment="outside [0, 1]",
    ),
    # Task 3 — happy paths
    SmokeCase("task3_perfect", 3, task3_perfect, expect_success=True),
    SmokeCase("task3_realistic", 3, task3_realistic, expect_success=True),
    SmokeCase("task3_random", 3, task3_random, expect_success=True),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_cli(task: int, preds_path: Path, gt_path: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        [
            "echorisk-score", "--task", str(task),
            "--predictions", str(preds_path),
            "--ground-truth", str(gt_path),
        ],
        capture_output=True, text=True, check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _run_case(case: SmokeCase, gt_by_task: dict[int, pd.DataFrame], tmpdir: Path) -> bool:
    gt = gt_by_task[case.task]
    rng = np.random.default_rng(abs(hash(case.name)) % (2**32))
    preds = case.predictions_builder(gt, rng)

    preds_path = tmpdir / f"{case.name}_predictions.csv"
    gt_path = tmpdir / f"task{case.task}_ground_truth.csv"
    preds.to_csv(preds_path, index=False)
    if not gt_path.exists():
        gt.to_csv(gt_path, index=False)

    rc, stdout, stderr = _run_cli(case.task, preds_path, gt_path)

    if case.expect_success:
        if rc != 0:
            print(f"  FAIL: expected success but CLI exited {rc}")
            print(f"    stderr: {stderr.strip()}")
            return False
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            print(f"  FAIL: stdout is not valid JSON: {stdout[:200]}")
            return False
        print(f"  OK: {_summarise_result(case.task, result)}")
        return True

    if rc == 0:
        print(f"  FAIL: expected rejection but CLI succeeded: {stdout[:200]}")
        return False
    if case.expected_error_fragment and case.expected_error_fragment not in stderr:
        print(f"  FAIL: expected error fragment '{case.expected_error_fragment}' not in stderr")
        print(f"    stderr: {stderr.strip()}")
        return False
    print(f"  OK: rejected with '{stderr.strip()}'")
    return True


def _summarise_result(task: int, result: dict) -> str:
    if task == 1:
        r_str = f"{result['pearson_r']:.3f}" if result['pearson_r'] is not None else "null"
        return (
            f"MAE={result['mae']:.3f}, RMSE={result['rmse']:.3f}, "
            f"r={r_str}, n={result['n']}"
        )
    if task == 2:
        return (
            f"AUROC={result['auroc']:.3f}, bal_acc={result['balanced_accuracy']:.3f}, "
            f"sens@spec90={result['sensitivity_at_90_specificity']:.3f}, "
            f"AUPRC={result['auprc']:.3f}, n={result['n']}, pos={result['n_positive']}"
        )
    if task == 3:
        return (
            f"AUROC={result['auroc']:.3f}, "
            f"sens@FPR10={result['sensitivity_at_fpr_10']:.3f}, "
            f"sens@FPR20={result['sensitivity_at_fpr_20']:.3f}, "
            f"Brier={result['brier']:.3f}, n={result['n']}, pos={result['n_positive']}"
        )
    raise ValueError(f"Unknown task {task}")


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    gt_by_task = {
        1: make_task1_gt(rng),
        2: make_task2_gt(rng),
        3: make_task3_gt(rng),
    }

    print("EchoRisk-MICCAI 2026 scoring smoke test")
    print("=" * 60)
    for task_id, gt in gt_by_task.items():
        n_patients = gt["patient_id"].nunique()
        n_exams = len(gt)
        print(
            f"Task {task_id} ground truth: {n_patients} patients, "
            f"{n_exams} exams, columns={list(gt.columns)}"
        )
    print()

    passed = failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for case in CASES:
            print(f"[task {case.task}] {case.name}")
            if _run_case(case, gt_by_task, tmpdir):
                passed += 1
            else:
                failed += 1
            print()

    print("=" * 60)
    print(f"Summary: {passed}/{len(CASES)} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())