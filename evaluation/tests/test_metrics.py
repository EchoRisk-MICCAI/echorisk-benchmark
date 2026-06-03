"""Tests for EchoRisk-MICCAI 2026 scoring metrics.

All three tasks use the composite (patient_id, timepoint) key.

Coverage targets:
- Happy path for each task: known predictions produce known metric values.
- Schema validation: missing columns, duplicate keys, non-matching keys.
- Range validation: out-of-range LVEF, out-of-range probabilities.
- Degenerate inputs: single-class ground truth, constant predictions.
- Sensitivity-at-FPR correctness on a hand-constructed ROC.
- Multi-timepoint aggregation: the same patient at different timepoints
  contributes as independent exams.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from echorisk_eval.metrics import (
    _sensitivity_at_fpr,
    score_task1,
    score_task2,
    score_task3,
)


def _gt_task1(patients: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build Task 1 ground truth from a list of (patient_id, timepoint, lvef)."""
    return pd.DataFrame(
        [
            {
                "patient_id": p,
                "timepoint": t,
                "video_a4c": f"{p}_{t}_A4C.dcm",  # realistic extra column
                "video_a2c": f"{p}_{t}_A2C.dcm",
                "lvef": v,
            }
            for p, t, v in patients
        ]
    )


def _pred_task1(
    predictions: list[tuple[str, str, float]], *, col: str = "lvef_pred"
) -> pd.DataFrame:
    """Build Task 1 predictions from a list of (patient_id, timepoint, pred)."""
    return pd.DataFrame(
        [{"patient_id": p, "timepoint": t, col: v} for p, t, v in predictions]
    )


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------


def test_task1_perfect_predictions_give_zero_error() -> None:
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 55.0), ("ECHORISK_0002", "T1", 40.0),
         ("ECHORISK_0003", "T1", 65.0)]
    )
    pred = _pred_task1(
        [("ECHORISK_0001", "T1", 55.0), ("ECHORISK_0002", "T1", 40.0),
         ("ECHORISK_0003", "T1", 65.0)]
    )
    result = score_task1(pred, gt)
    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.pearson_r == pytest.approx(1.0)
    assert result.n == 3


def test_task1_known_errors() -> None:
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 50.0), ("ECHORISK_0002", "T1", 60.0)]
    )
    pred = _pred_task1(
        [("ECHORISK_0001", "T1", 52.0), ("ECHORISK_0002", "T1", 58.0)]
    )
    result = score_task1(pred, gt)
    assert result.mae == pytest.approx(2.0)
    assert result.rmse == pytest.approx(2.0)


def test_task1_multiple_timepoints_same_patient() -> None:
    """Same patient at two timepoints contributes as two independent exams."""
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 60.0), ("ECHORISK_0001", "T2", 50.0),
         ("ECHORISK_0001", "T3", 55.0)]
    )
    pred = _pred_task1(
        [("ECHORISK_0001", "T1", 58.0), ("ECHORISK_0001", "T2", 52.0),
         ("ECHORISK_0001", "T3", 57.0)]
    )
    result = score_task1(pred, gt)
    assert result.mae == pytest.approx(2.0)
    assert result.n == 3


def test_task1_rejects_out_of_range_lvef() -> None:
    gt = _gt_task1([("ECHORISK_0001", "T1", 50.0)])
    pred = _pred_task1([("ECHORISK_0001", "T1", 150.0)])
    with pytest.raises(ValueError, match="LVEF range"):
        score_task1(pred, gt)


def test_task1_rejects_missing_prediction_column() -> None:
    gt = _gt_task1([("ECHORISK_0001", "T1", 50.0)])
    pred = pd.DataFrame(
        {"patient_id": ["ECHORISK_0001"], "timepoint": ["T1"], "wrong_col": [50.0]}
    )
    with pytest.raises(ValueError, match="missing required columns"):
        score_task1(pred, gt)


def test_task1_rejects_missing_timepoint_column() -> None:
    """Predictions without a timepoint column are rejected."""
    gt = _gt_task1([("ECHORISK_0001", "T1", 50.0)])
    pred = pd.DataFrame(
        {"patient_id": ["ECHORISK_0001"], "lvef_pred": [50.0]}
    )
    with pytest.raises(ValueError, match="missing required columns"):
        score_task1(pred, gt)


def test_task1_rejects_incomplete_coverage() -> None:
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 50.0), ("ECHORISK_0002", "T1", 60.0)]
    )
    pred = _pred_task1([("ECHORISK_0001", "T1", 50.0)])
    with pytest.raises(ValueError, match="must have a prediction"):
        score_task1(pred, gt)


def test_task1_rejects_duplicate_composite_keys() -> None:
    """Two rows with the same (patient_id, timepoint) are a hard error."""
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 50.0), ("ECHORISK_0002", "T1", 60.0)]
    )
    pred = _pred_task1(
        [("ECHORISK_0001", "T1", 50.0), ("ECHORISK_0001", "T1", 55.0)]
    )
    with pytest.raises(ValueError, match="duplicate"):
        score_task1(pred, gt)


def test_task1_same_patient_different_timepoints_not_duplicate() -> None:
    """Same patient_id at different timepoints is not a duplicate."""
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 55.0), ("ECHORISK_0001", "T2", 52.0)]
    )
    pred = _pred_task1(
        [("ECHORISK_0001", "T1", 55.0), ("ECHORISK_0001", "T2", 52.0)]
    )
    result = score_task1(pred, gt)
    assert result.n == 2


def test_task1_constant_predictions_give_nan_pearson() -> None:
    gt = _gt_task1(
        [("ECHORISK_0001", "T1", 50.0), ("ECHORISK_0002", "T1", 55.0),
         ("ECHORISK_0003", "T1", 60.0)]
    )
    pred = _pred_task1(
        [("ECHORISK_0001", "T1", 55.0), ("ECHORISK_0002", "T1", 55.0),
         ("ECHORISK_0003", "T1", 55.0)]
    )
    result = score_task1(pred, gt)
    assert np.isnan(result.pearson_r)
    assert result.mae == pytest.approx(10.0 / 3.0)


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------


def _gt_task2(exams: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": p,
                "timepoint": t,
                "video_a4c": f"{p}_{t}_A4C.dcm",
                "video_a2c": f"{p}_{t}_A2C.dcm",
                "lv_dysfunction": y,
            }
            for p, t, y in exams
        ]
    )


def _pred_task2(exams: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"patient_id": p, "timepoint": t, "dysfunction_pred": v}
            for p, t, v in exams
        ]
    )


def test_task2_perfect_predictions() -> None:
    gt = _gt_task2(
        [("ECHORISK_0001", "T1", 0), ("ECHORISK_0002", "T1", 1),
         ("ECHORISK_0003", "T1", 0), ("ECHORISK_0004", "T1", 1)]
    )
    pred = _pred_task2(
        [("ECHORISK_0001", "T1", 0.1), ("ECHORISK_0002", "T1", 0.9),
         ("ECHORISK_0003", "T1", 0.2), ("ECHORISK_0004", "T1", 0.8)]
    )
    result = score_task2(pred, gt)
    assert result.auroc == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.n == 4
    assert result.n_positive == 2


def test_task2_sensitivity_at_90_specificity_reported() -> None:
    """The sens@spec90 field is populated and in [0, 1]."""
    rng = np.random.default_rng(0)
    n = 50
    y_true = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    y_score = np.concatenate([rng.uniform(0.0, 0.5, n), rng.uniform(0.4, 1.0, n)])
    gt = _gt_task2([(f"p{i}", "T1", int(y)) for i, y in enumerate(y_true)])
    pred = _pred_task2([(f"p{i}", "T1", float(s)) for i, s in enumerate(y_score)])
    result = score_task2(pred, gt)
    assert 0.0 <= result.sensitivity_at_90_specificity <= 1.0


def test_task2_rejects_out_of_range_probability() -> None:
    gt = _gt_task2([("ECHORISK_0001", "T1", 0), ("ECHORISK_0002", "T1", 1)])
    pred = _pred_task2([("ECHORISK_0001", "T1", 0.5), ("ECHORISK_0002", "T1", 1.5)])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        score_task2(pred, gt)


def test_task2_rejects_single_class_ground_truth() -> None:
    gt = _gt_task2([("ECHORISK_0001", "T1", 0), ("ECHORISK_0002", "T1", 0)])
    pred = _pred_task2([("ECHORISK_0001", "T1", 0.3), ("ECHORISK_0002", "T1", 0.7)])
    with pytest.raises(ValueError, match="only one class"):
        score_task2(pred, gt)


def test_task2_rejects_non_binary_labels() -> None:
    gt = _gt_task2([("ECHORISK_0001", "T1", 0), ("ECHORISK_0002", "T1", 2)])
    pred = _pred_task2([("ECHORISK_0001", "T1", 0.3), ("ECHORISK_0002", "T1", 0.7)])
    with pytest.raises(ValueError, match="non-binary"):
        score_task2(pred, gt)


# ---------------------------------------------------------------------------
# Task 3 tests
# ---------------------------------------------------------------------------


def _gt_task3(exams: list[tuple[str, int]]) -> pd.DataFrame:
    """Task 3 is baseline-only (T1); helper enforces that."""
    return pd.DataFrame(
        [
            {
                "patient_id": p,
                "timepoint": "T1",
                "video_a4c": f"{p}_T1_A4C.dcm",
                "video_a2c": f"{p}_T1_A2C.dcm",
                "cardiotoxicity": y,
            }
            for p, y in exams
        ]
    )


def _pred_task3(exams: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"patient_id": p, "timepoint": "T1", "cardiotoxicity_pred": v}
            for p, v in exams
        ]
    )


def test_task3_perfect_discrimination() -> None:
    rng = np.random.default_rng(42)
    n = 50
    y_true = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    y_score = np.concatenate([rng.uniform(0.0, 0.4, n), rng.uniform(0.6, 1.0, n)])
    gt = _gt_task3([(f"p{i}", int(y)) for i, y in enumerate(y_true)])
    pred = _pred_task3([(f"p{i}", float(s)) for i, s in enumerate(y_score)])
    result = score_task3(pred, gt)
    assert result.auroc == pytest.approx(1.0)
    assert result.sensitivity_at_fpr_10 == pytest.approx(1.0)
    assert result.sensitivity_at_fpr_20 == pytest.approx(1.0)
    assert result.n == 2 * n
    assert result.n_positive == n


def test_task3_random_scores_give_auroc_near_half() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=2 * n)
    y_score = rng.uniform(0.0, 1.0, size=2 * n)
    gt = _gt_task3([(f"p{i}", int(y)) for i, y in enumerate(y_true)])
    pred = _pred_task3([(f"p{i}", float(s)) for i, s in enumerate(y_score)])
    result = score_task3(pred, gt)
    assert 0.4 < result.auroc < 0.6


# ---------------------------------------------------------------------------
# _sensitivity_at_fpr unit tests
# ---------------------------------------------------------------------------


def test_sensitivity_at_fpr_hand_crafted_roc() -> None:
    # 5 positives, 5 negatives. Sorted by score descending:
    # 0.9(+) 0.8(+) 0.7(+) 0.6(+) 0.5(-) 0.4(+) 0.3(-) 0.2(-) 0.1(-) 0.05(-)
    # At threshold 0.6: 4 TP, 0 FP -> TPR=0.8, FPR=0.0
    # At threshold 0.4: 5 TP, 1 FP -> TPR=1.0, FPR=0.2
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.5, 0.3, 0.2, 0.1, 0.05])
    assert _sensitivity_at_fpr(y_true, y_score, target_fpr=0.0) == pytest.approx(0.8)
    assert _sensitivity_at_fpr(y_true, y_score, target_fpr=0.2) == pytest.approx(1.0)
    assert _sensitivity_at_fpr(y_true, y_score, target_fpr=1.0) == pytest.approx(1.0)


def test_sensitivity_at_fpr_rejects_invalid_target() -> None:
    y_true = np.array([0, 1])
    y_score = np.array([0.3, 0.7])
    with pytest.raises(ValueError):
        _sensitivity_at_fpr(y_true, y_score, target_fpr=1.5)
    with pytest.raises(ValueError):
        _sensitivity_at_fpr(y_true, y_score, target_fpr=-0.1)