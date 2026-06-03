# EchoRisk-MICCAI 2026 Benchmark

Official evaluation code and reference implementations for the
[EchoRisk-MICCAI 2026 Challenge](https://echorisk-miccai.github.io) on
AI for cardiac function estimation, left ventricular dysfunction
assessment, and early prediction of therapy-induced cardiotoxicity from
echocardiography.

The challenge is a satellite event of
[MICCAI 2026](https://conferences.miccai.org/2026/) and is built on data
from the EU-funded [CARDIOCARE project](https://cardiocare-project.eu/)
(Horizon 2020, Grant No. 945175).

## Release schedule

This repository is under active development ahead of the challenge launch.

| Component                     | Location       | Status      |
|-------------------------------|----------------|-------------|
| Official scoring code         | `evaluation/`  | Released    |
| Metrics specification         | `docs/`        | Coming soon |
| Reference implementation code | `baselines/`   | July 2026   |

## Challenge overview

The challenge comprises three independent tasks on multicentre
echocardiography data collected from breast cancer patients undergoing
potentially cardiotoxic therapy:

- **Task 1 — LVEF regression.** Estimate left ventricular ejection
  fraction from 2D echocardiography video. Primary metric: MAE.
- **Task 2 — LV dysfunction classification.** Binary classification of
  left ventricular dysfunction based on a GLS threshold. Primary metric:
  AUC-ROC.
- **Task 3 — Early cardiotoxicity prediction.** Predict therapy-induced
  cardiotoxicity from baseline imaging only. Primary metric: AUC-ROC,
  with a clinical target of sensitivity ≥ 80% at FPR within 10–20%.

Teams may participate in one, two, or all three tasks. Rankings are
computed independently per task.

For data access, registration, full task specifications, and the
complete challenge timeline, see the
[challenge website](https://echorisk-miccai.github.io).

## Repository layout

```
echorisk-benchmark/
├── evaluation/     Official scoring code and tests
├── baselines/      Reference implementation code
└── docs/           Metrics specification and supporting documentation
```

Each subdirectory contains its own README describing contents and usage.

## Citation

If you use this code or the EchoRisk-MICCAI benchmark in your work, please cite:

```bibtex
@misc{echorisk_miccai_2026,
  title        = {EchoRisk-MICCAI 2026 Benchmark},
  author       = {EchoRisk-MICCAI 2026 Challenge Organisers},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19727928},
  url          = {https://doi.org/10.5281/zenodo.19727928}
}
```

## Licence

- Code in this repository is released under the [MIT Licence](LICENSE).
- Documentation is released under
  [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

- Challenge queries: **echorisk.miccai@gmail.com**
- Repository issues and bug reports: use the
  [GitHub issue tracker](https://github.com/EchoRisk-MICCAI/echorisk-benchmark/issues).

## Acknowledgements

This challenge is supported by the CARDIOCARE project, which has received
funding from the European Union's Horizon 2020 research and innovation
programme under Grant Agreement No. 945175.