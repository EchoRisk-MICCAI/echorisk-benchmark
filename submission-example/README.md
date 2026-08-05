# EchoRisk-MICCAI 2026 — Reference Submission Container

This directory is a **working, tested reference implementation** of the
container contract defined in
[`docs/HARNESS_SPEC.md`](../docs/HARNESS_SPEC.md). It does not perform
real inference; it is a dummy model that produces syntactically valid,
in-range predictions so you can verify your Docker plumbing, mounts, GPU
visibility, network isolation, manifest parsing, and output schema,
before wiring up your actual model.

**Read `docs/HARNESS_SPEC.md` first.** This README explains how to use
this reference container; the spec explains why the contract is shaped
this way and what the evaluation server actually does.

## Files

| File                    | Purpose                                                        |
|--------------------------|-------------------------------------------------------------------|
| `infer.py`               | Reference inference script. Replace `predict()` with your model.  |
| `entrypoint.sh`           | Container entrypoint. Do not modify.                              |
| `Dockerfile`              | Example image build. Adjust deps/weights, keep the `ENTRYPOINT`.  |
| `build.sh`                | Builds the image for a given task (`./build.sh <1\|2\|3>`).       |
| `run_local_test.sh`       | Runs the built image against `fixture/`, validates full coverage. |
| `fixture/manifest.csv`   | Synthetic manifest, includes both single-missing-view cases.      |
| `fixture/videos/`        | Placeholder files only, not real DICOM. See note below.           |

## Quick start

```bash
# Build a Task 1 image
./build.sh 1

# Run it against the local fixture (matches the real evaluation invocation)
./run_local_test.sh 1

# If your machine has no GPU (plumbing check only — not a substitute for
# testing under real GPU visibility before you submit):
NO_GPU=1 ./run_local_test.sh 1
```

A successful run ends with:

```
OK: output covers all 4 manifest rows.
```

Repeat for tasks 2 and 3 by changing the argument.

## Adapting this for your own submission

1. Copy this directory into your own project.
2. Implement your model loading and inference logic in `infer.py`,
   inside the `predict` function. Do not change:
   - The manifest-loading logic (`_load_manifest`, `_resolve_view`)
   - The per-row `try`/`except` structure in `main()` — this guarantees
     full row coverage even when individual cases fail (see §4 of the
     harness spec; this is a hard requirement, not a nicety)
   - The output CSV schema for your task
3. Add your model weights and any additional dependencies to the
   `Dockerfile`. Replace the base image if you need a different CUDA
   version or framework (PyTorch, TensorFlow, etc.) — just keep the
   `ENTRYPOINT` and the `/app` working directory.
4. Build and test locally against your own copy of the validation data
   before pushing to Synapse's Docker registry (see below — the bundled
   fixture is not suitable for this).

## Note on fixture data

`fixture/` contains synthetic placeholder files for testing container
plumbing only: mounts, manifest parsing, the nested per-patient,
per-timepoint video path convention, missing-view handling, and output
row coverage. **It does not contain real echocardiography data** and
cannot be used to validate your actual model's accuracy.

To test your real model's performance, run your container against your
own copy of the validation set (already available to you from Synapse),
using the same invocation pattern shown in `run_local_test.sh`, then
score the output with `echorisk-score` from the `evaluation/` package.

The fixture also includes a `T2` timepoint row for generality. Your real
Task 3 manifest will only ever contain `T1` rows (baseline-only); the
fixture's job is to test container plumbing, not enforce task-specific
data shape.

## Validating your output schema

Before submitting, confirm your container's output scores cleanly:

```bash
echorisk-score --task 1 \
    --predictions scratch_output/predictions.csv \
    --ground-truth /path/to/your/validation_gt.csv
```

Any `error:` about missing columns, duplicate keys, out-of-range values,
or incomplete coverage means your output does not yet match the
contract — fix it before submitting your image tag on Synapse.

## Submitting

1. Confirm your Synapse account is a Certified User (**Account Settings
   → Trust & Credentials**) — Docker push and submission fail silently
   otherwise, with no error explaining why.
2. Push your finished image to Synapse's Docker registry:
   ```bash
   docker login docker.synapse.org -u <your-synapse-username>
   docker build -t docker.synapse.org/<your-synapse-project-id>/<image-name>:<tag> .
   docker push docker.synapse.org/<your-synapse-project-id>/<image-name>:<tag>
   ```
3. Go to your own Synapse project's **Docker tab**, find your pushed
   image, and click **Docker Repository Tools → Submit Docker Repository
   to Challenge**. Select the evaluation queue matching your task.
4. Only one designated submitter per team has submission access — the
   Synapse account matching the Profile URL on your team's registration
   form. If that person hasn't accepted the Synapse Team invitation
   granting this access, submission is not possible until they do.

Each team has one final submission per task queue; only your last
submission before the deadline is evaluated. Submit only when you are
confident this is your final container. Full details:
[Docker Submission Guide](https://echorisk-miccai.github.io/submission-guide.html).