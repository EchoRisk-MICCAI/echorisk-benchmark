# EchoRisk-MICCAI 2026 — Metrics Specification

This document defines the evaluation metrics, ranking procedures, and
input validation rules used by the official scoring code in `evaluation/`.
Participants should read this alongside the
[evaluation README](../evaluation/README.md) to ensure their local scores
match the leaderboard exactly.

All metric functions are implemented in
`evaluation/src/echorisk_eval/metrics.py` and can be run locally via the
`echorisk-score` CLI or the Python API.

---

## Common conventions

- Every submission is a CSV file keyed by the composite identifier
  `(patient_id, timepoint)`.
- Predictions must cover **every** `(patient_id, timepoint)` pair present
  in the ground-truth file. Partial submissions are rejected with a
  non-zero exit code.
- If a Docker container fails to produce a valid output for one or more
  test cases, those cases are assigned the worst possible metric value for
  the corresponding task and metric. This policy is enforced at the
  submission harness level before the scoring code is invoked.
- All three tasks are evaluated **independently**. Participation in one
  task does not require participation in any other.

---

## Task 1 — Cardiac Parameter Estimation

### Clinical context

Participants estimate left ventricular ejection fraction (LVEF) directly
from 2D echocardiography video. LVEF is expressed as a percentage. The
clinically accepted measurement variability is approximately ±5 percentage
points; deviations beyond 10 percentage points may alter clinical
management around key cut-offs.

### Submission format

| Column       | Type   | Range     | Description                                  |
|--------------|--------|-----------|----------------------------------------------|
| `patient_id` | string | —         | Anonymised patient identifier                |
| `timepoint`  | string | T1 – T5   | Follow-up timepoint                          |
| `lvef_pred`  | float  | [0, 100]  | Predicted LVEF in percentage points          |

### Primary ranking metric

**Mean Absolute Error (MAE)**, in percentage points.

$$\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

Lower is better. Rankings are in ascending order of MAE.

### Secondary metrics (reported, not ranked)

- **Root Mean Square Error (RMSE)**

$$\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$

- **Pearson correlation coefficient** between predicted and reference
  values. Reported as `null` in JSON output when predictions are constant
  (undefined case).

### Tie-breaking

Ties in MAE (to three decimal places) are resolved by RMSE.

### Clinical interpretability metrics (reported only)

Agreement in detecting clinically relevant GLS deterioration (relative
reduction greater than 15% from baseline) is reported using sensitivity,
specificity, and F1-score. Binary labels are derived from reference and
predicted GLS values using this threshold. These metrics do not affect
ranking.

### Input validation

- `lvef_pred` must be in `[0, 100]`. Values outside this range are
  rejected.
- NaN values in any column are rejected.
- Duplicate `(patient_id, timepoint)` keys in either file are rejected.
- Predictions that do not cover all ground-truth exams are rejected.

---

## Task 2 — LV Dysfunction Assessment

### Clinical context

Participants classify left ventricular dysfunction from echocardiography
examinations. The positive class is defined as GLS ≥ −16% per ASE/EACVI
consensus (indicating dysfunction). Output must be a calibrated
probability, not a hard binary label.

### Submission format

| Column             | Type   | Range   | Description                              |
|--------------------|--------|---------|------------------------------------------|
| `patient_id`       | string | —       | Anonymised patient identifier            |
| `timepoint`        | string | T1 – T5 | Follow-up timepoint                      |
| `dysfunction_pred` | float  | [0, 1]  | Predicted probability of LV dysfunction  |

### Primary ranking metric

**Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**.

Higher is better. Rankings are in descending order of AUC-ROC. AUC-ROC
is computed using `sklearn.metrics.roc_auc_score`.

### Secondary metrics (reported, not ranked)

- **Sensitivity at 90% specificity** (first tie-breaker — see below).
  Computed as the maximum TPR on the ROC curve at any operating point
  with FPR ≤ 0.10.
- **Balanced accuracy** at threshold 0.5 (second tie-breaker).
- **AUPRC** (area under the precision-recall curve).
- **F1-score** at threshold 0.5.

### Tie-breaking

Ties in AUC-ROC (to three decimal places) are resolved sequentially:

1. Higher sensitivity at 90% specificity.
2. Higher balanced accuracy at threshold 0.5.

### Input validation

- `dysfunction_pred` must be in `[0, 1]`. Values outside this range are
  rejected.
- Ground-truth labels (`lv_dysfunction`) must be exactly 0 or 1.
- Ground truth containing only one class causes AUC-ROC to be undefined
  and is rejected.
- NaN values in any column are rejected.
- Duplicate `(patient_id, timepoint)` keys in either file are rejected.

---

## Task 3 — Early Prediction of Cardiotoxicity 

### Clinical context

Participants predict future therapy-induced cardiotoxicity from baseline
(T1) echocardiography alone, before any treatment-related cardiac changes
occur. The positive class is adjudicated cardiotoxicity within the
available follow-up window (up to 12 months). Output must be a calibrated
probability.

The intended clinical operating region is **sensitivity ≥ 80% at FPR
within 10–20%**, reflecting a realistic tolerance for increased
surveillance in cardio-oncology practice. Performance outside this region,
even if associated with high AUC, may not translate into clinically
actionable benefit.

### Submission format

| Column                | Type   | Range | Description                                     |
|-----------------------|--------|-------|-------------------------------------------------|
| `patient_id`          | string | —     | Anonymised patient identifier                   |
| `timepoint`           | string | T1    | Always T1 for Task 3 (retained for uniformity)  |
| `cardiotoxicity_pred` | float  | [0, 1]| Predicted probability of future cardiotoxicity  |

### Primary ranking metric

**Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**.

Higher is better. Rankings are in descending order of AUC-ROC.

### Secondary metrics (reported, not ranked)

- **Sensitivity at FPR 10%**: maximum TPR at any operating point with
  FPR ≤ 0.10.
- **Sensitivity at FPR 20%**: maximum TPR at any operating point with
  FPR ≤ 0.20.
- **Brier score**: mean squared difference between predicted probabilities
  and true binary labels. Lower is better. Used as third tie-breaker.
- **Balanced accuracy** at threshold 0.5.

### Tie-breaking

Ties in AUC-ROC (to three decimal places) are resolved sequentially:

1. Higher sensitivity at FPR 20%.
2. Higher balanced accuracy at threshold 0.5.
3. Lower Brier score.

### Input validation

- `cardiotoxicity_pred` must be in `[0, 1]`. Values outside this range
  are rejected.
- Ground-truth labels (`cardiotoxicity`) must be exactly 0 or 1.
- Ground truth containing only one class is rejected.
- NaN values in any column are rejected.
- Duplicate `(patient_id, timepoint)` keys in either file are rejected.

---

## Missing prediction policy

This policy is enforced by the **submission harness**, not the scoring
code. If a Docker container fails to produce a valid output for one or
more test cases, those cases are assigned the worst possible value for
each metric:

| Task   | Metric   | Worst-case value assigned |
|--------|----------|---------------------------|
| Task 1 | MAE      | Maximum possible error    |
| Task 2 | AUC-ROC  | 0.0                       |
| Task 3 | AUC-ROC  | 0.0                       |

The `echorisk-score` CLI itself requires a complete prediction file and
will reject partial submissions with exit code 1.

---

## JSON output format

The CLI prints a JSON object to stdout on success. All float values are
finite or `null` (`null` is used for undefined metrics such as Pearson r
when predictions are constant). Exit code is 0 on success, 1 on any
validation or scoring error. Errors are printed to stderr.

Example — Task 1:

```json
{
  "mae": 3.421,
  "rmse": 4.187,
  "pearson_r": 0.847,
  "n": 59
}
```

Example — Task 2:

```json
{
  "auroc": 0.821,
  "sensitivity_at_90_specificity": 0.643,
  "balanced_accuracy": 0.734,
  "auprc": 0.612,
  "f1": 0.589,
  "n": 421,
  "n_positive": 72
}
```

Example — Task 3:

```json
{
  "auroc": 0.774,
  "sensitivity_at_fpr_10": 0.512,
  "sensitivity_at_fpr_20": 0.731,
  "brier": 0.183,
  "balanced_accuracy": 0.698,
  "n": 76,
  "n_positive": 28
}
```

---

## Software and reproducibility

All metrics are computed using:

- `scikit-learn >= 1.3` (`roc_auc_score`, `balanced_accuracy_score`,
  `average_precision_score`, `brier_score_loss`, `f1_score`, `roc_curve`)
- `scipy >= 1.11` (`pearsonr`)
- `numpy >= 1.24`
- `pandas >= 2.0`

Exact versions are pinned in `evaluation/pyproject.toml`. To reproduce
leaderboard scores locally, install the package from the repository and
use the same `echorisk-score` CLI that runs on the evaluation server.

```bash
pip install -e "./evaluation[test]"
echorisk-score --task 1 --predictions preds.csv --ground-truth gt.csv
```