"""Metric implementations for the EchoRisk-MICCAI 2026 Challenge.

Three tasks, each with a primary ranking metric and secondary reporting metrics.
All metric functions take pandas DataFrames with a documented schema and return
a frozen dataclass of results. No bootstrap confidence intervals in v1.0; those
are planned for v1.1.

Primary ranking metrics (these determine the leaderboard):
- Task 1 (LVEF regression): MAE
- Task 2 (LV dysfunction classification): AUC-ROC
- Task 3 (cardiotoxicity prediction): AUC-ROC

All three tasks are scored at the exam level, keyed by the composite
(patient_id, timepoint) identifier. A single patient may contribute multiple
exams across timepoints T1-T5 for Tasks 1 and 2; Task 3 uses baseline (T1)
exams only but retains the same schema for uniformity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Prediction CSV schemas — enforced on input.
# All tasks: columns [patient_id, timepoint, <prediction>]
# Task 1: prediction is lvef_pred, a float in [0, 100].
# Task 2: prediction is dysfunction_pred, a probability in [0, 1].
# Task 3: prediction is cardiotoxicity_pred, a probability in [0, 1].
# Ground-truth CSVs use the same (patient_id, timepoint) key with
# appropriate label columns.

_PATIENT_ID_COL: Final[str] = "patient_id"
_TIMEPOINT_COL: Final[str] = "timepoint"
_KEY_COLS: Final[tuple[str, str]] = (_PATIENT_ID_COL, _TIMEPOINT_COL)


# ---------------------------------------------------------------------------
# Task 1: LVEF regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task1Result:
    """Task 1 scoring result. Primary ranking metric: MAE.

    RMSE and Pearson r are reported as secondary metrics for context.
    """

    mae: float  # primary
    rmse: float
    pearson_r: float
    n: int


def score_task1(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    pred_col: str = "lvef_pred",
    true_col: str = "lvef",
) -> Task1Result:
    """Score Task 1: LVEF regression from 2D echocardiography video.

    Args:
        predictions: DataFrame with columns [patient_id, timepoint, lvef_pred].
        ground_truth: DataFrame with columns [patient_id, timepoint, lvef].
            Additional columns in ground_truth (e.g. video filenames) are
            ignored.
        pred_col: Name of the prediction column (default: "lvef_pred").
        true_col: Name of the ground-truth column (default: "lvef").

    Returns:
        Task1Result with MAE (primary), RMSE, Pearson r, and exam count.

    Raises:
        ValueError: If required columns are missing, values are out of the
            valid LVEF range [0, 100], or no exams match between inputs.
    """
    merged = _merge_and_validate(
        predictions, ground_truth, pred_col=pred_col, true_col=true_col
    )
    _validate_lvef_range(merged[pred_col].to_numpy(), name="predictions")
    _validate_lvef_range(merged[true_col].to_numpy(), name="ground truth")

    y_pred = merged[pred_col].to_numpy(dtype=np.float64)
    y_true = merged[true_col].to_numpy(dtype=np.float64)

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    # Pearson r is undefined when either input is constant; guard explicitly.
    if np.std(y_pred) == 0.0 or np.std(y_true) == 0.0:
        pearson_r = float("nan")
    else:
        pearson_r = float(pearsonr(y_pred, y_true).statistic)

    return Task1Result(mae=mae, rmse=rmse, pearson_r=pearson_r, n=len(merged))


# ---------------------------------------------------------------------------
# Task 2: LV dysfunction classification (GLS-based binary label)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task2Result:
    """Task 2 scoring result. Primary ranking metric: AUC-ROC.

    Tie-breaking:
      1. Higher sensitivity at 90% specificity.
      2. Higher overall accuracy at the LVEF < 50% clinical threshold (threshold 0.5).
    """

    auroc: float  # primary
    sensitivity_at_90_specificity: float  # first tie-breaker
    balanced_accuracy: float  # second tie-breaker (accuracy at threshold 0.5)
    auprc: float
    f1: float
    n: int
    n_positive: int


def score_task2(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    pred_col: str = "dysfunction_pred",
    true_col: str = "lv_dysfunction",
    threshold: float = 0.5,
) -> Task2Result:
    """Score Task 2: LV dysfunction binary classification.

    Positive class: GLS >= -16% per ASE/EACVI consensus (indicating dysfunction).

    Args:
        predictions: DataFrame with columns
            [patient_id, timepoint, dysfunction_pred]. dysfunction_pred is the
            predicted probability of the positive class.
        ground_truth: DataFrame with columns
            [patient_id, timepoint, lv_dysfunction]. lv_dysfunction is 0 or 1.
            Additional columns in ground_truth are ignored.
        pred_col: Name of the prediction column.
        true_col: Name of the ground-truth column.
        threshold: Probability threshold for balanced accuracy and F1.

    Returns:
        Task2Result with AUROC (primary), balanced accuracy and
        sensitivity-at-90%-specificity (secondary), and additional metrics.

    Raises:
        ValueError: If required columns are missing, probabilities are outside
            [0, 1], labels are not binary, or the ground truth contains only
            one class (which makes AUROC undefined).
    """
    merged = _merge_and_validate(
        predictions, ground_truth, pred_col=pred_col, true_col=true_col
    )
    _validate_probability_range(merged[pred_col].to_numpy(), name=pred_col)
    _validate_binary_labels(merged[true_col].to_numpy(), name=true_col)

    y_score = merged[pred_col].to_numpy(dtype=np.float64)
    y_true = merged[true_col].to_numpy(dtype=np.int64)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Task 2 ground truth contains only one class; AUROC/AUPRC undefined."
        )

    y_pred_binary = (y_score >= threshold).astype(np.int64)
    # Sensitivity at 90% specificity == sensitivity at FPR <= 0.10.
    sens_at_spec_90 = _sensitivity_at_fpr(y_true, y_score, target_fpr=0.10)

    return Task2Result(
        auroc=float(roc_auc_score(y_true, y_score)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred_binary)),
        sensitivity_at_90_specificity=sens_at_spec_90,
        auprc=float(average_precision_score(y_true, y_score)),
        f1=float(f1_score(y_true, y_pred_binary, zero_division=0)),
        n=len(merged),
        n_positive=int(y_true.sum()),
    )


# ---------------------------------------------------------------------------
# Task 3: Early cardiotoxicity prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task3Result:
    """Task 3 scoring result. Primary ranking metric: AUC-ROC.

    Clinical target: sensitivity >= 0.80 at FPR within 0.10-0.20.
    """

    auroc: float  # primary
    sensitivity_at_fpr_10: float
    sensitivity_at_fpr_20: float
    brier: float
    balanced_accuracy: float  # at threshold 0.5
    n: int
    n_positive: int


def score_task3(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    pred_col: str = "cardiotoxicity_pred",
    true_col: str = "cardiotoxicity",
    threshold: float = 0.5,
) -> Task3Result:
    """Score Task 3: early cardiotoxicity prediction from baseline imaging.

    All ground-truth rows are baseline (timepoint = T1), but the timepoint
    column is retained in the schema for uniformity with Tasks 1 and 2.

    Args:
        predictions: DataFrame with columns
            [patient_id, timepoint, cardiotoxicity_pred].
            cardiotoxicity_pred is the predicted probability of the positive class.
        ground_truth: DataFrame with columns
            [patient_id, timepoint, cardiotoxicity]. cardiotoxicity is 0 or 1.
            Additional columns in ground_truth are ignored.
        pred_col: Name of the prediction column.
        true_col: Name of the ground-truth column.
        threshold: Probability threshold for balanced accuracy.

    Returns:
        Task3Result with AUROC (primary), sensitivity at FPR 0.10 and 0.20,
        Brier score, balanced accuracy, exam count, and positive-class count.

    Raises:
        ValueError: If required columns are missing, probabilities are outside
            [0, 1], labels are not binary, or the ground truth contains only
            one class.
    """
    merged = _merge_and_validate(
        predictions, ground_truth, pred_col=pred_col, true_col=true_col
    )
    _validate_probability_range(merged[pred_col].to_numpy(), name=pred_col)
    _validate_binary_labels(merged[true_col].to_numpy(), name=true_col)

    y_score = merged[pred_col].to_numpy(dtype=np.float64)
    y_true = merged[true_col].to_numpy(dtype=np.int64)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Task 3 ground truth contains only one class; AUROC undefined."
        )

    y_pred_binary = (y_score >= threshold).astype(np.int64)

    return Task3Result(
        auroc=float(roc_auc_score(y_true, y_score)),
        sensitivity_at_fpr_10=_sensitivity_at_fpr(y_true, y_score, target_fpr=0.10),
        sensitivity_at_fpr_20=_sensitivity_at_fpr(y_true, y_score, target_fpr=0.20),
        brier=float(brier_score_loss(y_true, y_score)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred_binary)),
        n=len(merged),
        n_positive=int(y_true.sum()),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_and_validate(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    pred_col: str,
    true_col: str,
) -> pd.DataFrame:
    """Inner-join predictions and ground truth on (patient_id, timepoint).

    Validates schema, duplicates, and complete coverage of the ground-truth set.
    Returns a DataFrame with the key columns plus pred_col and true_col.
    """
    for df, name, required in (
        (predictions, "predictions", {*_KEY_COLS, pred_col}),
        (ground_truth, "ground_truth", {*_KEY_COLS, true_col}),
    ):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    key_cols = list(_KEY_COLS)
    if predictions.duplicated(subset=key_cols).any():
        raise ValueError(
            "predictions contains duplicate (patient_id, timepoint) entries."
        )
    if ground_truth.duplicated(subset=key_cols).any():
        raise ValueError(
            "ground_truth contains duplicate (patient_id, timepoint) entries."
        )

    # Restrict ground_truth to key + label columns before merging to avoid
    # pulling along video filename columns and similar. Restrict predictions
    # similarly so the merged frame only contains the columns we need.
    gt_subset = ground_truth[[*key_cols, true_col]]
    pred_subset = predictions[[*key_cols, pred_col]]

    merged = pred_subset.merge(
        gt_subset, on=key_cols, how="inner", validate="one_to_one"
    )
    if len(merged) == 0:
        raise ValueError(
            "No (patient_id, timepoint) overlap between predictions and ground truth."
        )
    if len(merged) < len(ground_truth):
        # This is a hard failure, not a warning, because the policy for missing
        # predictions is defined at the submission harness level (worst-case
        # assignment per BIAS spec), not at the metric level. Metrics operate
        # only on complete prediction sets.
        raise ValueError(
            f"predictions cover {len(merged)} of {len(ground_truth)} "
            f"ground-truth exams; all exams must have a prediction."
        )
    return merged


def _validate_lvef_range(values: np.ndarray, *, name: str) -> None:
    """LVEF must be in [0, 100] as a percentage."""
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN values.")
    if np.any(values < 0.0) or np.any(values > 100.0):
        raise ValueError(
            f"{name} contains values outside the valid LVEF range [0, 100]."
        )


def _validate_probability_range(values: np.ndarray, *, name: str) -> None:
    """Probabilities must be in [0, 1]."""
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} contains values outside [0, 1].")


def _validate_binary_labels(values: np.ndarray, *, name: str) -> None:
    """Binary labels must be exactly 0 or 1."""
    if np.any(np.isnan(values.astype(np.float64))):
        raise ValueError(f"{name} contains NaN values.")
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"{name} contains non-binary values: {sorted(unique)}; "
            "expected exactly {0, 1}."
        )


def _sensitivity_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, *, target_fpr: float
) -> float:
    """Maximum TPR on the ROC curve at any operating point with FPR <= target_fpr.

    Uses sklearn.metrics.roc_curve to determine operating points, which handles
    tie-breaking consistently. If no operating point satisfies the FPR
    constraint, returns 0.0.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}.")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid_points = fpr <= target_fpr
    return float(tpr[valid_points].max()) if valid_points.any() else 0.0