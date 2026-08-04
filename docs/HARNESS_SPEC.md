# EchoRisk-MICCAI 2026 — Container Harness Specification

This document is the exact contract your Docker container must satisfy for
test-set evaluation. Read this in full before building your image. A
working reference container implementing this contract is provided in
[`submission-example/`](../submission-example/).

## 1. Invocation

The evaluation server runs your container **once**, with no arguments,
against the complete test manifest. There is no per-case relaunch and no
harness-side result aggregation: your container is solely responsible for
producing one output file covering every case in the manifest before it
exits.

Exact command (not illustrative — this is what runs):

```bash
docker run --rm \
    --gpus '"device=0"' \
    --network none \
    --memory 40g \
    -v /path/to/test/input:/input:ro \
    -v /path/to/scratch/output:/output \
    <image>:<tag>
```

- `--gpus '"device=0"'` pins a single, specific GPU device index. Your code
  must not enumerate devices or implement multi-GPU selection logic. Assume
  exactly one visible device and use it directly (`cuda:0` / device index 0
  from inside the container).
- `--network none` disables all network access, including DNS resolution.
  Any code path that calls out to the internet (weight downloads, API
  calls, telemetry) will fail. Bundle every dependency and weight file
  inside the image at build time.
- `--memory 40g` caps your container's system RAM. The evaluation server
  has 48 GB total; 40 GB is made available per container run, leaving
  headroom for the host OS and evaluation harness process. Design for this
  ceiling, not the full 48 GB.
- No arguments are passed. Do not require CLI flags, stdin input, or
  environment variables at run time.
- `-v /path/to/test/input:/input:ro` mounts the test data into the container at /input, read-only (:ro). /path/to/test/input is a placeholder for wherever the real test data lives on the evaluation server; you never see or control that path — from inside your container, it always appears at /input, exactly as documented in §2. The :ro means your container cannot modify or delete the test data, only read it. 
- `-v /path/to/scratch/output:/output` mounts a writable scratch directory into the container at /output. This is where your container must write predictions.csv. Like the input path, /path/to/scratch/output is server-side and opaque to you; from inside the container, you always write to /output, and that's the only location the evaluation server checks afterward.
- Your `ENTRYPOINT` must exit with code `0` on success. A non-zero exit
  code fails the entire submission for that task.

## 2. Input contract

Mounted read-only at `/input`.

```
/input/
├── manifest.csv
└── videos/
    ├── ECHORISK_0001/
    │   └── T1/
    │       ├── ECHORISK_0001_T1_A4C.dcm
    │       └── ECHORISK_0001_T1_A2C.dcm
    ├── ECHORISK_0002/
    │   ├── T1/
    │   │   └── ECHORISK_0002_T1_A4C.dcm
    │   └── T2/
    │       ├── ECHORISK_0002_T2_A4C.dcm
    │       └── ECHORISK_0002_T2_A2C.dcm
    └── ...
```

Video files are organised **one subfolder per patient, then one subfolder
per timepoint**, under `/input/videos/`: `{patient_id}/{timepoint}/{file}`.
This is a three-level nested structure, not a flat file listing and not a
single level of per-patient nesting. Do not assume all DICOM files for a
patient sit in one folder together across timepoints.

### `manifest.csv` schema

| Column        | Type   | Description                                                   |
|---------------|--------|-------------------------------------------------------------------|
| `patient_id`  | string | Anonymised patient identifier                                   |
| `timepoint`   | string | Follow-up timepoint (`T1`–`T5`; always `T1` for Task 3)          |
| `video_a4c`   | string | Path relative to `/input/videos/`, or empty string if absent     |
| `video_a2c`   | string | Path relative to `/input/videos/`, or empty string if absent     |

`video_a4c` and `video_a2c` are **relative paths, not bare filenames**
(e.g. `ECHORISK_0003/T1/ECHORISK_0003_T1_A4C.dcm`). Resolve them by
joining directly with `/input/videos/`; do not assume a flat directory,
do not assume a single level of nesting, and do not reconstruct the path
from `patient_id` and `timepoint` yourself — use the manifest value
exactly as given, since it already contains the full relative path.

Label columns present in the training/validation manifests (e.g. `lvef`,
`biomarker_elevated`) are **stripped entirely** from the test manifest,
not blanked. Do not write code that expects those columns to exist, even
as empty values, in the test-time manifest.

### Missing-view convention

A missing echocardiography view is represented as an **empty string** in
the corresponding column. There is no file to check for absence:
**presence in the manifest is authoritative.** Do not attempt to detect a
missing view by checking whether a file exists on disk; check whether the
manifest cell is empty instead.

A row may have:
- Both `video_a4c` and `video_a2c` populated.
- Exactly one of the two populated, the other an empty string.
- In rare cases, both empty (see §4, row-coverage requirement — your
  container must still produce a prediction for this row).

## 3. Output contract

Your container must write **exactly one file**:

```
/output/predictions.csv
```

This is the file consumed directly by the official scorer
(`echorisk-score`, package `echorisk_eval`) with no intermediate
aggregation step. The schema is task-specific and must match exactly:

| Task | Output column           | Type  | Range     |
|------|---------------------------|-------|-----------|
| 1    | `lvef_pred`                | float | [0, 100]  |
| 2    | `dysfunction_pred`         | float | [0, 1]    |
| 3    | `cardiotoxicity_pred`      | float | [0, 1]    |

Every row also requires `patient_id` and `timepoint`, matching the
manifest exactly.

## 4. Row-coverage requirement (read this carefully)

**Your container must write a prediction row for every `(patient_id,
timepoint)` pair present in the manifest, with no exceptions.**

This is stricter than "worst-case metric assignment on failure" might
suggest. The official scorer (`echorisk_eval.metrics`) enforces complete
coverage as a hard precondition: if your output file is missing even one
row that appears in the ground-truth set, the scorer raises a validation
error and **the entire submission fails to score**, not just the missing
case.

Practically, this means:

- If inference on a case raises an exception (corrupt DICOM, both views
  missing, out-of-memory, anything), your code must catch it and still
  write a row for that case using a defensive fallback value (e.g. your
  training-set prior, or a fixed clinically neutral value such as 55%
  LVEF or a 0.5 probability).
- Do not let a per-case exception propagate up and skip the row.
- Do not filter out rows you were unable to process.

The reference implementation in `submission-example/infer.py` demonstrates
this pattern: every case is wrapped in a `try`/`except` that falls back to
a constant prediction on failure, guaranteeing full coverage regardless of
per-case errors.

## 5. Task identity

Your image is built for a **single task**. There is no runtime flag or
environment variable indicating which task is being scored; the
evaluation server invokes your image with no arguments, so task identity
must be resolved before run time.

Bake the task ID into the image at build time using a Docker build
argument:

```bash
docker build --build-arg TASK_ID=1 -t <image>:<tag> .
```

If you are participating in multiple tasks, build and submit a **separate
image per task**, each to its corresponding Synapse evaluation queue.

## 6. Image naming and tagging

Name and tag your image as:

```
<synapse-team-name>-task<N>:<submission-tag>
```

Example: `cardiovision-lab-task2:final`

Use a descriptive tag for `<submission-tag>` (e.g. `final`, `v2`) rather
than `latest`, since the evaluation server pulls the exact tag you submit
on Synapse and `latest` invites accidental overwrites between your
development pushes and your actual final submission.

## 7. Operational constraints

These apply regardless of what is described above; restated here so you
do not need to cross-reference the challenge website.

**Minimum requirements for your own development/testing:**

- One GPU with at least 16 GB VRAM.
- At least 32 GB system RAM.

**Confirmed evaluation server specification** (what your submitted image
actually runs on):

- GPU: NVIDIA A40, 48 GB VRAM.
- CPU: 32 cores.
- System RAM: 48 GB total, 40 GB made available per container run
  (see `--memory 40g` in §1).
- Storage: 1 TB, shared across the evaluation pipeline (not a per-team
  allocation).

Design and test against the minimum requirements above so your submission
is robust regardless of exact hardware; the confirmed specification is
provided so you know the ceiling you actually have to work with on
submission.

**Other constraints:**

- **No internet access** at inference time. `--network none` is enforced
  by the evaluation server; design and test against this locally (see
  `submission-example/run_local_test.sh`).
- **No manual intervention.** Your container must run fully automatically
  from `docker run` to exit.
- **Wall-clock limit: 15 minutes per task per case.** The clock starts
  when your container process starts (i.e. from `docker run`, after the
  image has already been pulled) and includes any model-loading or
  initialisation time inside your `ENTRYPOINT`, not just active inference.
  The evaluation harness times your container against the largest case in
  the test set first. Exceeding the limit on that case terminates the
  submission for the task. Profile your total container run time,
  including startup, against your largest local validation-set case
  before submitting.
- **All model weights and dependencies must be baked into the image at
  build time.** Nothing may be fetched at run time.
- **One resubmission window per task.** You may rebuild and resubmit
  before the deadline; only your last submission per task queue is
  evaluated.

## 8. Failure modes and what they cost you

| Failure                                                        | Consequence                                      |
|------------------------------------------------------------------|---------------------------------------------------|
| Container exits non-zero                                         | Entire task submission fails, no ranking          |
| `/output/predictions.csv` missing or empty                       | Entire task submission fails, no ranking          |
| Output file missing one or more manifest rows                    | Entire task submission fails to score (§4)         |
| Output value out of the valid range for the task                 | Rejected by the scorer as a validation error       |
| Wall-clock exceeded on the largest case                          | Submission terminated, treated as a failed run     |
| Individual case fails inside your code but row is still written  | That row's prediction is scored normally; if it's a poor fallback value, it costs you accuracy on that case only, not the whole submission |

The last row is the entire point of §4: a defensive fallback on a bad case
costs you one data point. A missing row costs you the whole task.