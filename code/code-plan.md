# Paper 1 - Code and simulation plan

Rule: anything that computes a reported number is Python. MATLAB renders only.

## Layout
src/
  00_ingest.py      raw survey files -> parquet, no edits
  01_audit.py       LOD table, censoring rates, discordant pairs   [E0a, E1c]
  02_simulate.py    synthetic dose-response                        [E0b]
  03_masks.py       mask, survival indicator, mask-density         [E1a,b,d]
  04_transform.py   raw / CLR / ILR
  05_regimes.py     LOD/2, deletion, mask-feature, common floor    [E2]
  06_models.py      XGBoost reg + AFT, logistic, baselines
  07_validate.py    random CV, spatial block, LOSO
  08_stats.py       bootstrap CIs, McNemar, DeLong
  09_report.py      tables + figure data
conf/               one YAML per experiment
data/               raw -> interim -> processed
out/                tables, figures, logs

## Rules
- python src/NN_x.py --config conf/x.yaml. No notebooks in the pipeline.
- Config carries the seed. Written beside every output.
- Nothing downstream of 01_audit runs until the gate passes.
- run_all.sh reproduces every number in the paper.
- MATLAB reads out/tables/*.csv for figures. Never writes back.

## Build order
1. 00 + 01 - the gate. Stop until AUC and discordant-pair counts are in.
2. 02 - simulation. No data dependency. Decides whether the mechanism is causal.
3. 03 + 04 + 05 + 06 - main run.
4. 07 + 08 + 09.
Steps 1 and 2 are ~300 lines. Write those and nothing else first.

## Interfaces
- Row carries: survey, x, y, concentrations, mask, LOD used.
- Mask is a separate array, never merged into the concentration matrix.
- AFT labels: label_lower_bound = -inf, label_upper_bound = LOD, per row.

## Simulation spec (02_simulate.py)
- k synthetic surveys from ONE generative process. Geochemistry identical by
  construction.
- Sweep inter-survey LOD disparity 0-95% induced censoring, 5% steps.
- 100 repetitions per step.
- Measure: mask->survey AUC, and cross-survey task inflation.
- Output: mean with 95% CI vs disparity.
- Flat curve = mechanism not real. Stop the project.
