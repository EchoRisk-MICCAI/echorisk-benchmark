# EchoRisk-MICCAI 2026 — Evaluation

Official scoring code for the three challenge tasks.

This package implements the metrics used on the challenge leaderboard.
Participants are encouraged to use the same code locally to score their
predictions on the validation set during model development, so that
local numbers match the leaderboard numbers exactly.

## Installation

Requires Python 3.10 or newer. From the repository root:

```bash
pip install -e "./evaluation[test]"
```

This installs the `echorisk_eval` package and its dependencies
(numpy, pandas, scikit-learn, scipy) along with the test suite.
The `echorisk-score` command becomes available on your PATH.

## Command-line usage

Score a submission for any of the three tasks:

```bash
echorisk-score --task 1 --predictions preds.csv --ground-truth gt.csv
echorisk-score --task 2 --predictions preds.csv --ground-truth gt.csv
echorisk-score --task 3 --predictions preds.csv --ground-truth gt.csv
```

Output is a JSON dictionary of metric values printed to stdout:

```json
{
  "mae": 3.421,
  "rmse": 4.187,
  "pearson_r": 0.847,
  "n": 59
}
```

Exit code is 0 on success, 1 on validation or scoring errors.
Errors are printed to stderr with a descriptive message.

## Python API

Use the scoring functions directly in your training or evaluation scripts:

```python
import pandas as pd
from echorisk_eval import score_task1, score_task2, score_task3

predictions = pd.read_csv("my_predictions.csv")
ground_truth = pd.read_csv("validation_labels.csv")

result = score_task1(predictions, ground_truth)
print(f"MAE: {result.mae:.3f}")
print(f"Pearson r: {result.pearson_r:.3f}")
```

Each `score_task*` function returns a frozen dataclass
(`Task1Result`, `Task2Result`, `Task3Result`) with the primary and
secondary metrics. Primary ranking metrics are documented in the
class docstrings.

## Prediction file format

All submissions are CSV files keyed by the composite identifier
`(patient_id, timepoint)`. A single patient may contribute multiple
exams across timepoints T1-T5 for Tasks 1 and 2; Task 3 uses baseline
(T1) exams only but retains the same schema for uniformity.

Predictions must cover every exam in the ground-truth set; partial
submissions are rejected.

### Task 1 — LVEF regression

| Column       | Type   | Range    | Description                                  |
|--------------|--------|----------|----------------------------------------------|
| `patient_id` | string | —        | Anonymised patient identifier                |
| `timepoint`  | string | T1–T5    | Follow-up timepoint (baseline to 12 months)  |
| `lvef_pred`  | float  | [0, 100] | Predicted left ventricular ejection fraction |

Example:

```csv
patient_id,timepoint,lvef_pred
ECHORISK_0002,T1,70.8
ECHORISK_0002,T2,66.5
ECHORISK_0003,T1,41.2
ECHORISK_0004,T1,54.8
```

### Task 2 — LV dysfunction classification

| Column              | Type   | Range  | Description                                       |
|---------------------|--------|--------|---------------------------------------------------|
| `patient_id`        | string | —      | Anonymised patient identifier                     |
| `timepoint`         | string | T1–T5  | Follow-up timepoint                               |
| `dysfunction_pred`  | float  | [0, 1] | Predicted probability of LV dysfunction           |

Submissions must output calibrated probabilities, not hard labels.
The positive class corresponds to GLS >= -16% per ASE/EACVI consensus.

Example:

```csv
patient_id,timepoint,dysfunction_pred
ECHORISK_0003,T1,0.23
ECHORISK_0004,T1,0.81
ECHORISK_0008,T2,0.45
ECHORISK_0010,T1,0.15
ECHORISK_0010,T2,0.38
ECHORISK_0010,T4,0.72
```

### Task 3 — Early cardiotoxicity prediction

| Column                 | Type   | Range  | Description                                    |
|------------------------|--------|--------|------------------------------------------------|
| `patient_id`           | string | —      | Anonymised patient identifier                  |
| `timepoint`            | string | T1     | Baseline (always T1 for Task 3)                |
| `cardiotoxicity_pred`  | float  | [0, 1] | Predicted probability of future cardiotoxicity |

Submissions must output calibrated probabilities derived from
baseline echocardiography only. The `timepoint` column is always
`T1` for Task 3 but is retained in the schema for uniformity with
Tasks 1 and 2.

Example:

```csv
patient_id,timepoint,cardiotoxicity_pred
ECHORISK_0006,T1,0.15
ECHORISK_0007,T1,0.72
```

## Metrics

### Task 1 — LVEF regression

- **Primary (ranking):** Mean Absolute Error (MAE), in percentage points.
  Lower is better.
- **Secondary:** Root Mean Square Error (RMSE), Pearson correlation
  coefficient.

### Task 2 — LV dysfunction classification

- **Primary (ranking):** Area under the receiver operating characteristic
  curve (AUC-ROC). Higher is better.
- **Secondary (tie-breaking and clinical interpretation):** balanced
  accuracy at threshold 0.5, sensitivity at 90% specificity, AUPRC, F1.

### Task 3 — Early cardiotoxicity prediction

- **Primary (ranking):** Area under the receiver operating characteristic
  curve (AUC-ROC). Higher is better.
- **Secondary (clinical interpretation):** sensitivity at FPR 10% and
  FPR 20%, Brier score, balanced accuracy at threshold 0.5.
- **Clinical target:** sensitivity >= 80% at FPR within 10–20%
  (safe screening operating point).

Precise mathematical definitions of each metric are given in the
metrics specification document released with the dataset.

## Running the tests

From the `evaluation/` directory:

```bash
pytest
```

The test suite covers metric correctness, input validation, multi-timepoint
aggregation, and edge cases (constant predictions, single-class ground
truth, invalid ranges, schema violations, missing timepoint column).

## Validation policy

The scoring code enforces strict input validation:

- **Missing exams:** predictions must cover every `(patient_id, timepoint)`
  exam present in the ground-truth set. Partial submissions are rejected
  with an error.
- **Out-of-range values:** LVEF predictions outside [0, 100] and
  probability predictions outside [0, 1] are rejected.
- **Duplicate composite keys:** each `(patient_id, timepoint)` pair must
  appear exactly once in both files. The same patient at different
  timepoints is not a duplicate.
- **Missing columns:** required columns must be present with the exact
  names documented above. In particular, the `timepoint` column is
  required in all three tasks.

The submission-harness policy for handling missing predictions on the
test set (worst-case metric assignment per BIAS specification) is
applied by the submission infrastructure before this code is invoked.
The metrics themselves operate on complete, validated prediction sets.

## Licence

MIT — see the [LICENSE](../LICENSE) file at the repository root.