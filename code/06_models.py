"""
06_models - the downstream task, the model zoo, and an honest cost estimate

This script does not run the experiment. It defines it, verifies every piece on
one target, and measures what the full grid in 07_validate.py will cost before
anyone commits a night to it.

THE TASK

Leave-one-element-out prediction, after Zhang et al. (2021): one shared element is
held out and predicted from the other 43. The target rotates across all 44, which
is what makes the uncertainty in 08_stats.py a bootstrap over elements rather than
over an element someone chose.

Rotation is also the design's own falsification test. Elements are stratified by
their cross-survey detection-limit ratio, and if censoring drives the inflation
then inflation must rise with that ratio. If it does not, the mechanism is refuted
on real data whatever 03_masks.py returned.

WHAT IS HELD IDENTICAL ACROSS REGIMES

The scored rows. A regime cannot be credited for predicting a value it invented,
so a test row counts only when the target was genuinely measured - uncensored in
the delivered data AND above the common floor. That set is computed once, from the
original mask and the floor, and reused for every regime and every representation.
Without this, regime (a) would be scored partly on its own substituted constants,
which are trivially predictable, and it would win by construction.

FEATURES BY REGIME

  (a) substitution     non-detects replaced by half of each survey's own limit.
  (c) mask_feature     the values of (a) plus 44 binary mask columns.
  (d) common_floor     everything below the common floor replaced by half of that
                       floor - one constant per element, identical in all three
                       surveys, which is what removes the survey-dependence from
                       the feature values. The target is handled differently: it
                       is left censored and passed to the AFT objective as an
                       interval, never substituted.
  (b) complete_case    absent by construction. 05_regimes.py found it retains
                       three NASGL and two NGSA samples out of 6,125, so it cannot
                       be validated across surveys at all.

REPRESENTATION

Applied after the regime fills values, never before, because a log-ratio is
undefined at zero and transforming first would force a substitution before the
comparison of substitutions has been made. The contrast matrix is loaded from
02-datasets/processed/, not rebuilt, so that every stage shares one geometry.

The log-ratio arms carry log10 of the total as an extra coordinate, so that raw,
CLR and ILR encode identical information and the comparison between them is about
geometry rather than about what each one silently discards.

MODELS

  xgb_regression     squared error on log10 concentration. The workhorse.
  xgb_aft            survival:aft, with label_lower_bound = 0 and
                     label_upper_bound = the limit on censored targets. This is
                     the only model that can use a censored target as an
                     observation rather than as a number.
  ridge              a linear model on the same coordinates, so that any claim
                     about gradient boosting is made against a model a reader can
                     audit.
  survey_mean        predict each survey's own training mean. Beats many published
                     models under leave-one-survey-out and is reported for that
                     reason.
  coordinates_only   latitude and longitude, nothing else, after Ploton et al.
                     (2020). If this matches the full model, the full model has
                     learned position.
  mask_density_only  the per-row count of non-detects, nothing else. If this
                     matches the full model, the full model has learned the
                     laboratory.

Outputs to 05-results/
    06_model_smoke.csv     every model x regime x representation on one target
    06_models_report.txt   including the measured cost of the full grid

Usage
    python 06_models.py
    python 06_models.py --target Cu --folds 5 --repeats 5
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, '..', '02-datasets', 'processed')
RESULTS = os.path.join(HERE, '..', '05-results')
SEED = 20260906

REGIMES = ['substitution', 'mask_feature', 'common_floor']
REPRESENTATIONS = ['raw', 'clr', 'ilr']


def require_xgboost():
    if importlib.util.find_spec('xgboost') is None:
        sys.exit('xgboost is not installed, and the censored objective has no '
                 'substitute in scikit-learn.\n'
                 '  pip install xgboost')
    import xgboost
    return xgboost


def read_table(base, folder=PROCESSED):
    for extension, reader in (('.parquet', pd.read_parquet), ('.csv.gz', pd.read_csv)):
        path = os.path.join(folder, base + extension)
        if os.path.exists(path):
            return reader(path)
    sys.exit(f'missing input: {base} in {folder}. Run 05_regimes.py first.')


# ------------------------------------------------------------------ the data
def load_task_inputs():
    """Everything the task needs, with the scored-row set fixed once and for all."""
    meta = read_table('regime_base_meta')
    substituted = read_table('regime_substitution_conc')
    mask = read_table('regime_base_mask')
    floored_mask = read_table('regime_common_floor_mask')
    bounds = read_table('regime_common_floor_bounds')
    elements = list(substituted.columns)

    floor = pd.read_csv(os.path.join(RESULTS, '05_common_floor.csv')) \
        .set_index('element').common_floor.reindex(elements)

    psi = pd.read_csv(os.path.join(PROCESSED, 'ilr_contrast_matrix.csv'), index_col=0)
    if list(psi.columns) != elements:
        sys.exit('the contrast matrix and the regime matrices disagree on element '
                 'order. One of them was rebuilt without the other; re-run '
                 '04_transform.py and 05_regimes.py in order.')

    # The common-floor feature matrix: one substitution constant per element, the
    # same number in every survey, which is exactly what the regime is for.
    half_floor = (floor / 2.0).to_numpy()[None, :]
    floored_values = substituted.where(
        floored_mask == 0,
        pd.DataFrame(np.broadcast_to(half_floor, substituted.shape),
                     columns=elements, index=substituted.index))

    values = {'substitution': substituted,
              'mask_feature': substituted,
              'common_floor': floored_values}
    return meta, values, mask, floored_mask, bounds, floor, elements, psi.to_numpy()


def scoreable_rows(mask, floored_mask, target):
    """Test rows a regime is allowed to be scored on.

    Uncensored in the delivered data and above the common floor, so the number
    being predicted is a number somebody measured. Identical for every regime,
    which is the only way the comparison between regimes means anything.
    """
    return ((mask[target] == 0) & (floored_mask[target] == 0)).to_numpy()


# -------------------------------------------------------- representation
def centred_log_ratio(values):
    logged = np.log(np.asarray(values, dtype=float))
    return logged - logged.mean(axis=1, keepdims=True)


def represent(values, kind, psi):
    """Raw means untransformed; the log-ratio arms carry the total alongside.

    A log-ratio transformation deletes the size of the composition and keeps only
    its shape. The target here is an absolute concentration, so a compositional
    arm without the size is being asked to predict a magnitude from ratios alone
    while the raw arm keeps it - and the comparison then measures that omission
    rather than the geometry. The first version of this script did exactly that
    and made the log-ratio arms look catastrophic.

    Appending log10 of the total restores it. Shape plus size is an invertible
    reparameterisation of the raw vector, so all three arms hold the SAME
    information and differ only in the coordinates they express it in, which is
    the only comparison worth reporting.
    """
    array = np.asarray(values, dtype=float)
    if kind == 'raw':
        return array
    total = np.log10(array.sum(axis=1, keepdims=True))
    if kind == 'clr':
        return np.hstack([centred_log_ratio(array), total])
    if kind == 'ilr':
        return np.hstack([centred_log_ratio(array) @ psi.T, total])
    raise ValueError(kind)


def build_design(values, mask, meta, target, elements, regime, representation, psi):
    """Features and target for one rotation of the leave-one-element-out task.

    The target column is removed from the features before any transformation.
    Removing it afterwards would leave it inside every log-ratio coordinate, and
    the model would be predicting a quantity from itself.
    """
    predictors = [e for e in elements if e != target]
    reduced = psi_for(psi, elements, predictors) if representation == 'ilr' else None
    features = represent(values[predictors], representation, reduced)
    if regime == 'mask_feature':
        features = np.hstack([features, mask[predictors].to_numpy(dtype=float)])
    y = np.log10(values[target].to_numpy(dtype=float))
    return features, y


def psi_for(psi, elements, predictors):
    """Rebuild the contrast matrix for the reduced part set.

    Dropping the target part changes the composition, so the frozen 43 x 44 matrix
    no longer applies. The balances involving the dropped part are removed and the
    remainder re-orthonormalised by Gram-Schmidt, which keeps the frozen partition
    as the source of the coordinates instead of inventing a new one per target.
    """
    keep = np.array([elements.index(p) for p in predictors])
    reduced = psi[:, keep]
    rows = []
    for row in reduced:
        row = row - row.mean()
        for existing in rows:
            row = row - existing * float(row @ existing)
        norm = np.linalg.norm(row)
        if norm > 1e-10:
            rows.append(row / norm)
        if len(rows) == len(predictors) - 1:
            break
    return np.vstack(rows)


# ------------------------------------------------------------------- models
def fit_predict(name, train, test, features, y, meta, mask, bounds, target, xgb):
    """One model, one split. Returns predictions on log10 concentration."""
    if name == 'survey_mean':
        frame = pd.DataFrame({'survey': meta.survey.to_numpy()[train], 'y': y[train]})
        means = frame.groupby('survey').y.mean()
        overall = float(np.nanmean(y[train]))
        # Under leave-one-survey-out the held-out survey is absent from training,
        # so this falls back to the pooled mean. That is not a flaw in the
        # baseline; it is what any model faces when it meets a new region.
        return np.array([means.get(s, overall)
                         for s in meta.survey.to_numpy()[test]])

    if name == 'coordinates_only':
        columns = meta[['latitude', 'longitude']].to_numpy(dtype=float)
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 tree_method='hist', random_state=0, n_jobs=-1)
        model.fit(np.nan_to_num(columns[train]), y[train])
        return model.predict(np.nan_to_num(columns[test]))

    if name == 'mask_density_only':
        density = mask.sum(axis=1).to_numpy(dtype=float)[:, None]
        model = Ridge(alpha=1.0)
        model.fit(density[train], y[train])
        return model.predict(density[test])

    if name == 'ridge':
        model = Ridge(alpha=1.0)
        model.fit(features[train], y[train])
        return model.predict(features[test])

    if name == 'xgb_regression':
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 tree_method='hist', random_state=0, n_jobs=-1)
        model.fit(features[train], y[train])
        return model.predict(features[test])

    if name == 'xgb_aft':
        # The censored target enters as an interval, not as a number. A censored
        # row says only that the concentration lies somewhere below the limit,
        # which is exactly what a lower bound of zero and an upper bound at the
        # limit expresses, and is the whole reason this objective is here.
        concentration = np.power(10.0, y)
        upper = bounds[target].to_numpy(dtype=float)
        censored = np.isfinite(upper)
        lower = np.where(censored, 0.0, concentration)
        upper = np.where(censored, upper, concentration)

        matrix = xgb.DMatrix(features[train])
        matrix.set_float_info('label_lower_bound', lower[train])
        matrix.set_float_info('label_upper_bound', upper[train])
        booster = xgb.train({'objective': 'survival:aft',
                             'eval_metric': 'aft-nloglik',
                             'aft_loss_distribution': 'normal',
                             'aft_loss_distribution_scale': 1.0,
                             'tree_method': 'hist',
                             'max_depth': 6,
                             'eta': 0.1,
                             'seed': 0},
                            matrix, num_boost_round=200)
        predicted = booster.predict(xgb.DMatrix(features[test]))
        return np.log10(np.clip(predicted, 1e-12, None))

    raise ValueError(name)


def r_squared(truth, predicted):
    """Unbounded below, which is the point: a model can be worse than the mean."""
    residual = np.nansum((truth - predicted) ** 2)
    total = np.nansum((truth - np.nanmean(truth)) ** 2)
    return float(1.0 - residual / total) if total > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description='Task, models, and grid cost')
    ap.add_argument('--target', default='Cu',
                    help='element to smoke-test on; the full run rotates over all')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    xgb = require_xgboost()
    os.makedirs(RESULTS, exist_ok=True)

    (meta, values, mask, floored_mask, bounds, floor,
     elements, psi) = load_task_inputs()
    if args.target not in elements:
        sys.exit(f'{args.target} is not one of the {len(elements)} shared elements')

    surveys = sorted(meta.survey.unique())
    print(f'{len(meta):,} rows, {len(elements)} elements, target {args.target}')

    scoreable = scoreable_rows(mask, floored_mask, args.target)
    print(f'scoreable rows for {args.target}: {int(scoreable.sum()):,} '
          f'({100 * scoreable.mean():.1f}%)')

    models = ['survey_mean', 'coordinates_only', 'mask_density_only', 'ridge',
              'xgb_regression', 'xgb_aft']
    rows = []
    for regime in REGIMES:
        representations = REPRESENTATIONS if regime == 'substitution' else ['ilr']
        for representation in representations:
            features, y = build_design(values[regime], mask, meta, args.target,
                                       elements, regime, representation, psi)
            for name in models:
                if name == 'xgb_aft' and regime != 'common_floor':
                    continue  # a censored objective needs a censored target
                start = time.perf_counter()
                scores = []
                for held_out in surveys:
                    test = (meta.survey == held_out).to_numpy()
                    train = ~test
                    predicted = fit_predict(name, train, test, features, y, meta,
                                            mask, bounds, args.target, xgb)
                    keep = scoreable[test]
                    scores.append(r_squared(y[test][keep], np.asarray(predicted)[keep]))
                elapsed = time.perf_counter() - start
                rows.append({'target': args.target, 'regime': regime,
                             'representation': representation, 'model': name,
                             'n_features': features.shape[1],
                             'r2_loso_median': float(np.nanmedian(scores)),
                             **{f'r2_holdout_{s}': round(v, 4)
                                for s, v in zip(surveys, scores)},
                             'seconds_for_3_fits': round(elapsed, 2)})
                print(f'  {regime:14} {representation:4} {name:18} '
                      f'R2 LOSO median {rows[-1]["r2_loso_median"]:+.3f}  '
                      f'{elapsed:.1f}s')

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, '06_model_smoke.csv'), index=False)

    # What 07_validate.py will cost, measured rather than guessed.
    seconds_per_fit = float((table.seconds_for_3_fits / 3.0).sum())
    fits_per_target = args.folds * args.repeats + len(surveys)
    grid_hours = seconds_per_fit * fits_per_target * len(elements) / 3600.0

    lines = [
        '06_models report',
        '',
        f'rows              {len(meta):,}',
        f'elements          {len(elements)}   target rotates over all of them',
        f'smoke target      {args.target}',
        f'scoreable rows    {int(scoreable.sum()):,} '
        f'({100 * scoreable.mean():.1f}%) - uncensored AND above the common floor',
        '',
        'ONE TARGET, LEAVE-ONE-SURVEY-OUT',
        '  ' + table[['regime', 'representation', 'model', 'n_features',
                      'r2_loso_median'] + [f'r2_holdout_{s}' for s in surveys]]
        .to_string(index=False).replace('\n', '\n  '),
        '',
        '  Every regime is scored on the SAME rows: those where the target was',
        '  measured and sits above the common floor. A regime scored on its own',
        '  substituted constants would be predicting a number it wrote itself.',
        '',
        '  Negative R2 is not a bug. Leave-one-survey-out asks a model trained on',
        '  two continents to predict a third, and a model can be worse than that',
        f'  continent\'s own mean. Compare against survey_mean in the table above,',
        '  not against zero.',
        '',
        'READ THE BASELINES FIRST',
        '  coordinates_only and mask_density_only are adversarial. If either comes',
        '  close to xgb_regression, the full model has learned position or',
        '  laboratory rather than geochemistry, and nothing downstream can repair',
        '  that reading.',
        '',
        'COST OF THE FULL GRID',
        f'  measured                {seconds_per_fit:.2f} s per fit, summed over '
        f'the model set above',
        f'  fits per target         {fits_per_target} '
        f'({args.folds} folds x {args.repeats} repeats + {len(surveys)} '
        f'leave-one-survey-out)',
        f'  targets                 {len(elements)}',
        f'  estimated total         {grid_hours:.1f} hours, before spatial blocking',
        '',
        '  Spatial blocking adds one fold set per survey and is costed in',
        '  07_validate.py, where the block size comes from the variogram rather',
        '  than from a guess.',
        '',
        'written to 05-results/: 06_model_smoke.csv, 06_models_report.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '06_models_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
