# detection-limit-leakage

Code, results and figures for:

> **Detection-Limit Censoring Inflates Reported Cross-Region Generalization in Pooled Geochemical Surveys: Limited Mitigation by Spatial Cross-Validation**

Every survey reports concentrations against detection limits chosen by its own
laboratory, so the pattern of non-detects belongs to the analytical protocol as much
as to the rock. When surveys are pooled to train a model, that pattern becomes a
legible marker of which survey a sample came from. This repository contains the
pipeline that measures what the marker costs.

Headline results, all reproducible from `code/`:

| | |
|---|---|
| Censoring pattern classifies survey (no concentrations used) | 0.978 balanced accuracy, chance 0.333 |
| Random minus leave-one-survey-out R², primary arm | +0.589 [+0.414, +0.771] |
| Share of that gap recovered by spatial blocking | 0.7 % – 7.9 % |
| Inflation vs cross-survey detection-limit ratio | Spearman ρ = +0.379 [+0.135, +0.594] |
| Cost of harmonising to a common floor | 16,138 resolved values discarded in GEMAS |

---

## Data is not in this repository

The three surveys are third-party releases under their own terms and are not
redistributed here. Obtain them from the original providers:

| Survey | Source |
|---|---|
| NASGL (USA, A horizon) | USGS Data Series 801, Appendix 3 |
| NGSA (Australia) | Geoscience Australia Record 2011/020 |
| GEMAS (Europe, Ap aqua regia) | EuroGeoSurveys Geochemistry Expert Group |
| PWGD v2.3c / v2.3n | USGS Produced Waters Geochemical Database |

Place them under `02-datasets/` following the layout in `code/code-plan.md`, then
run the pipeline from `code/`. `00_ingest.py` reports exactly what it found and
fails loudly rather than guessing.

---

## Running the pipeline

Scripts run in numbered order. Each writes to `05-results/` and prints a report
that states its own verdict, including when that verdict is a failure.

```
00_ingest.py              raw survey files -> aligned concentration, mask and limit tables
01_audit.py               detection-limit audit, discordant-pair test          [E0a, E1c]
02_simulate.py            dose-response simulation, 2,000 repetitions          [E0b]
03_masks.py               mask as an identity channel, survival indicator      [E1a, E1b, E1d]
03_figure_dose_response.py    Figure 1
04_transform.py           pooled composition, frozen log-ratio geometry
05_regimes.py             four censoring-handling regimes, variable screen     [E2]
06_models.py              task definition, model zoo, measured grid cost
07_pipeline_selftest.py   pre-flight: run this before the grid
07_validate.py            three validation protocols, checkpointed             [E3]
08_stats.py               bootstrap intervals, paired tests, falsification test
09_report.py              tables and figure data
10_figures.py             Figures 2 and 3
```

Two notes on cost. `07_validate.py` takes roughly 5.6 hours for 44 targets and
checkpoints every fold, so an interrupted run resumes with the same command.
`07_pipeline_selftest.py` takes minutes and should pass before you start it.

Requires Python 3.10+, `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`,
`matplotlib`, and `pyarrow` if you want Parquet rather than gzipped CSV.

---

## What is checked, and how

The pre-flight suite (`07_pipeline_selftest.py`) tests the validation machinery
rather than the science, because a long run that finishes is not a run that is
correct:

- the target never reaches the model as a feature, verified by correlation
- two identical fits agree to the bit
- a permuted target returns R² that is not materially positive
- no spatial block appears on both sides of a fold
- leave-one-survey-out leaves no part of the held-out survey in training
- every regime is scored on identical rows
- the checkpoint contains no conflicting duplicates

---

## Reading the results honestly

Several outputs run against the paper's own framing and are kept as they are:

- the censoring mask predicts **survey** at 0.978 and **concentration** at +0.016,
  so the identity signal travels through substituted values rather than through the
  mask supplied as a feature;
- the pre-registered primary representation (isometric log-ratio) is outperformed
  under leave-one-survey-out by raw concentrations;
- the within-survey discordant-pair test (E1c) fails, and is reported as a negative
  result with its mechanism rather than removed;
- complete-case deletion is reported as infeasible at 43 predictors, not compared.

Three reports also record design corrections made after seeing results — a GEMAS
mask rule that discarded the most heavily censored elements, an E1d feature
selection that was circular, and an acceptance gate that was unsatisfiable by
construction. They are in the scripts' docstrings and in the printed reports.

---

## Citation

`CITATION.cff` in this repository. Cite the tagged release, not `main`.

## Licence

MIT for everything under `code/`. CC BY 4.0 for `results/` and `figures/`.
