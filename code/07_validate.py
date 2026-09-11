"""
07_validate - the same models under three validation protocols, and the gap

The paper's claim lives in the difference between protocols, not in any single
score. A pooled random split and a leave-one-survey-out split are asked the same
question of the same fitted model on the same data; the distance between their
answers is how much of the reported performance was never generalisation at all.

  random    pooled k-fold across all surveys. Reported as the inflated reference,
            not as a performance estimate. 02_simulate.py showed this protocol is
            blind to detection-limit disparity by construction: it stayed flat at
            R2 = 0.32 while leave-one-survey-out fell to -1.25 on identical data.
  blocked   spatially blocked k-fold, block size taken from the fitted variogram
            range of the covariates and never chosen by hand.
  loso      leave-one-survey-out. The distribution of scientific interest for any
            claim that a model transfers between regions.

No protocol is declared correct. Ploton et al. (2020) report collapse from random
to spatial; Tziachris et al. (2023) report a modest effect the other way; Wadoux
et al. (2021) argue spatial cross-validation lacks a theoretical basis for map
accuracy. Reporting one of them would take a side silently, so all three are
reported for every regime, every representation and every model, after Hengl et
al. (2026).

INFLATION

  inflation = R2(random) - R2(loso)

per target, regime, representation and model. Elements are stratified by their
cross-survey detection-limit ratio, and the design's own falsification test is
whether inflation rises with that ratio. If it does not, the mechanism is refuted
on real data whatever the mask experiments returned - which is why this script
computes the relationship rather than assuming it.

AREA OF APPLICABILITY

Reported for the leave-one-survey-out folds using the importance-weighted
dissimilarity index of Meyer and Pebesma (2021). A cross-region score means
nothing without knowing whether the held-out region was inside the space the
model was trained on, and reporting the share that falls outside is the
difference between a failed model and an extrapolating one.

COST AND RESUMPTION

The grid is hours long, so every fold is written as it completes and a restart
skips what is already on disk. Baselines that do not read the feature matrix -
the per-survey mean, the coordinates-only model, the mask-density-only model -
are fitted once per regime rather than once per representation, because fitting
them three times with identical inputs would be pure waste and would also invite
the reader to think they differed.

Outputs to 05-results/
    07_validation_folds.csv    one row per fold, the checkpoint and the evidence
    07_validation_summary.csv  aggregated, with inflation per combination
    07_spatial_blocks.csv      fitted variogram range and block size per survey
    07_area_of_applicability.csv
    07_validate_report.txt

Usage
    python 07_validate.py
    python 07_validate.py --targets Cu,Zn,Pb --folds 5 --repeats 5
    python 07_validate.py --protocols random,loso
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '05-results')
SEED = 20260906
EARTH_RADIUS_KM = 6371.0

FEATURE_MODELS = ['ridge', 'xgb_regression']
BASELINE_MODELS = ['survey_mean', 'coordinates_only', 'mask_density_only']
PRIMARY = ('substitution', 'ilr', 'xgb_regression')
CHECKPOINT_KEY = ['target', 'regime', 'representation', 'model', 'protocol', 'fold']


def load_models_module():
    """06_models.py holds the task definition; it is loaded rather than copied so
    that the two scripts cannot drift apart."""
    path = os.path.join(HERE, '06_models.py')
    if not os.path.exists(path):
        sys.exit(f'missing {path}')
    spec = importlib.util.spec_from_file_location('models_06', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ geography
def to_local_km(latitude, longitude):
    """Equirectangular projection about the survey's own centre.

    Good to a fraction of a percent over a single continent and, unlike a global
    projection, it distorts nothing at the scale block sizes are chosen at. Each
    survey is projected about its own centre because they are on three different
    continents and no single tangent plane serves all of them.
    """
    latitude = np.radians(np.asarray(latitude, dtype=float))
    longitude = np.radians(np.asarray(longitude, dtype=float))
    centre = np.nanmean(latitude)
    x = EARTH_RADIUS_KM * (longitude - np.nanmean(longitude)) * np.cos(centre)
    y = EARTH_RADIUS_KM * (latitude - centre)
    return np.column_stack([x, y])


def exponential_variogram(distance, nugget, sill, autocorrelation_range):
    return nugget + (sill - nugget) * (1.0 - np.exp(-distance / autocorrelation_range))


def fit_autocorrelation_range(coordinates, values, rng, n_sample=1200, n_bins=15):
    """Empirical semivariogram, exponential fit, return the range in km.

    Returns NaN rather than a fallback when the fit fails. A block size invented
    because a fit did not converge would be exactly the hand-chosen block size
    this protocol exists to avoid.
    """
    finite = np.isfinite(values) & np.isfinite(coordinates).all(axis=1)
    if finite.sum() < 100:
        return np.nan
    index = np.flatnonzero(finite)
    if len(index) > n_sample:
        index = rng.choice(index, size=n_sample, replace=False)
    points, z = coordinates[index], values[index]

    separation = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    semivariance = 0.5 * (z[:, None] - z[None, :]) ** 2
    upper = np.triu_indices(len(index), k=1)
    separation, semivariance = separation[upper], semivariance[upper]

    limit = np.percentile(separation, 50)
    inside = separation <= limit
    if inside.sum() < 100:
        return np.nan
    edges = np.linspace(0, limit, n_bins + 1)
    which = np.clip(np.digitize(separation[inside], edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    averaged = np.array([semivariance[inside][which == b].mean() if (which == b).any()
                         else np.nan for b in range(n_bins)])
    usable = np.isfinite(averaged)
    if usable.sum() < 5:
        return np.nan
    try:
        guess = [averaged[usable][0], averaged[usable][-1], limit / 3.0]
        fitted, _ = curve_fit(exponential_variogram, centres[usable], averaged[usable],
                              p0=guess, maxfev=20000,
                              bounds=([0, 0, 1e-3], [np.inf, np.inf, limit * 5]))
    except (RuntimeError, ValueError):
        return np.nan
    fitted_range = float(fitted[2])
    # A range at or beyond the window the variogram was fitted over means the
    # semivariance never reached a sill inside the data: the optimiser ran to its
    # bound and returned a number, not a range. Accepting it would set the block
    # size to the size of the continent and quietly turn blocked validation back
    # into random validation, which is the one outcome this protocol must not
    # produce silently.
    if not np.isfinite(fitted_range) or fitted_range >= limit:
        return np.nan
    return fitted_range


def spatial_blocks(meta, features, elements, rng, n_covariates=8):
    """Block size per survey from the median covariate autocorrelation range.

    The median over several covariates rather than one, because a single element
    can have a range set by one geological province, and the block size then
    describes that province instead of the data.
    """
    rows, block_id = [], np.full(len(meta), -1, dtype=np.int64)
    offset = 0
    for survey in sorted(meta.survey.unique()):
        index = np.flatnonzero((meta.survey == survey).to_numpy())
        coordinates = to_local_km(meta.latitude.to_numpy()[index],
                                  meta.longitude.to_numpy()[index])
        chosen = rng.choice(features.shape[1],
                            size=min(n_covariates, features.shape[1]), replace=False)
        ranges = [fit_autocorrelation_range(coordinates, features[index, c], rng)
                  for c in chosen]
        ranges = [r for r in ranges if np.isfinite(r)]
        size = float(np.median(ranges)) if ranges else np.nan
        rows.append({'survey': survey, 'n_covariates_fitted': len(ranges),
                     'range_km_median': round(size, 2) if np.isfinite(size) else np.nan,
                     'range_km_min': round(min(ranges), 2) if ranges else np.nan,
                     'range_km_max': round(max(ranges), 2) if ranges else np.nan})
        if not np.isfinite(size) or size <= 0:
            continue
        grid = np.floor(coordinates / size).astype(np.int64)
        keys = grid[:, 0] * 100003 + grid[:, 1]
        _, compact = np.unique(keys, return_inverse=True)
        block_id[index] = compact + offset
        offset += compact.max() + 1
        rows[-1]['n_blocks'] = int(compact.max() + 1)
    return block_id, pd.DataFrame(rows)


# ------------------------------------------------------------------ protocols
def protocol_splits(name, meta, block_id, folds, repeats, seed):
    """Yield (fold label, train mask, test mask) for one protocol."""
    n = len(meta)
    if name == 'loso':
        for survey in sorted(meta.survey.unique()):
            test = (meta.survey == survey).to_numpy()
            yield f'holdout_{survey}', ~test, test
        return
    if name == 'random':
        splitter = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        for i, (train, test) in enumerate(splitter.split(np.arange(n))):
            train_mask = np.zeros(n, bool); train_mask[train] = True
            yield f'fold_{i}', train_mask, ~train_mask
        return
    if name == 'blocked':
        usable = block_id >= 0
        splitter = GroupKFold(n_splits=folds)
        positions = np.flatnonzero(usable)
        for i, (train, test) in enumerate(
                splitter.split(positions, groups=block_id[usable])):
            train_mask = np.zeros(n, bool); train_mask[positions[train]] = True
            test_mask = np.zeros(n, bool); test_mask[positions[test]] = True
            yield f'block_fold_{i}', train_mask, test_mask
        return
    raise ValueError(name)


def area_of_applicability(features, train, test, weights=None):
    """Meyer and Pebesma (2021) dissimilarity index, and the share inside it.

    Each test point's distance to its nearest training point, in scaled and
    importance-weighted feature space, divided by the mean nearest-neighbour
    distance WITHIN the training set. The threshold is the 95th percentile of the
    training set's own index, so the question asked is whether a test point is
    further from the training data than the training data is from itself.
    """
    scaled = np.asarray(features, dtype=float)
    centre = np.nanmean(scaled[train], axis=0)
    spread = np.nanstd(scaled[train], axis=0)
    spread[spread == 0] = 1.0
    scaled = (scaled - centre) / spread
    if weights is not None:
        scaled = scaled * np.asarray(weights, dtype=float)[None, :]
    scaled = np.nan_to_num(scaled)

    neighbours = NearestNeighbors(n_neighbors=2).fit(scaled[train])
    within, _ = neighbours.kneighbors(scaled[train])
    reference = float(within[:, 1].mean())
    if reference <= 0:
        return np.nan, np.nan
    index_train = within[:, 1] / reference
    threshold = float(np.percentile(index_train, 95))
    outward, _ = neighbours.kneighbors(scaled[test], n_neighbors=1)
    index_test = outward[:, 0] / reference
    return float((index_test <= threshold).mean() * 100.0), float(np.median(index_test))


# ---------------------------------------------------------------- the grid
def combinations(elements, targets, regimes):
    """Every unit of work, with baselines attached to the regime rather than to
    the representation, since they never read the feature matrix."""
    plan = []
    for target in targets:
        for regime in regimes:
            representations = ['raw', 'clr', 'ilr'] if regime == 'substitution' \
                else ['ilr']
            for representation in representations:
                models = list(FEATURE_MODELS)
                if regime == 'common_floor':
                    models.append('xgb_aft')
                for model in models:
                    plan.append((target, regime, representation, model))
            for model in BASELINE_MODELS:
                plan.append((target, regime, 'none', model))
    return plan


def load_checkpoint(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=CHECKPOINT_KEY), set()
    done = pd.read_csv(path, float_precision='round_trip')
    keys = set(map(tuple, done[CHECKPOINT_KEY].astype(str).to_numpy()))
    return done, keys


def main():
    ap = argparse.ArgumentParser(description='Three protocols, one grid')
    ap.add_argument('--targets', default='all')
    ap.add_argument('--regimes', default='substitution,mask_feature,common_floor')
    ap.add_argument('--protocols', default='random,blocked,loso')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--restart', action='store_true',
                    help='ignore the checkpoint and recompute everything')
    args = ap.parse_args()

    models = load_models_module()
    xgb = models.require_xgboost()
    os.makedirs(RESULTS, exist_ok=True)

    (meta, values, mask, floored_mask, bounds, floor,
     elements, psi) = models.load_task_inputs()
    targets = elements if args.targets == 'all' else args.targets.split(',')
    unknown = [t for t in targets if t not in elements]
    if unknown:
        sys.exit(f'not shared elements: {unknown}')
    regimes = args.regimes.split(',')
    protocols = args.protocols.split(',')

    rng = np.random.default_rng(SEED)
    reference_features, _ = models.build_design(
        values['substitution'], mask, meta, targets[0], elements,
        'substitution', 'clr', psi)
    block_id, block_table = spatial_blocks(meta, reference_features, elements, rng)
    block_table.to_csv(os.path.join(RESULTS, '07_spatial_blocks.csv'), index=False)
    print(block_table.to_string(index=False))
    if 'blocked' in protocols:
        if (block_id >= 0).sum() == 0:
            sys.exit('no variogram reached a sill inside the data, so no block size '
                     'exists. Rather than invent one, drop "blocked" from '
                     '--protocols and report that spatial blocking was not '
                     'defensible on these surveys.')
        thin = block_table[block_table.get('n_blocks', pd.Series(dtype=float))
                           .fillna(0) < args.folds]
        if len(thin):
            sys.exit('these surveys yield fewer blocks than folds, so a blocked '
                     'split would put whole surveys in one fold:\n'
                     f'{thin.to_string(index=False)}\n'
                     f'Lower --folds below the smallest block count, or drop '
                     '"blocked" from --protocols.')

    checkpoint_path = os.path.join(RESULTS, '07_validation_folds.csv')
    if args.restart and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    previous, done = load_checkpoint(checkpoint_path)
    plan = combinations(elements, targets, regimes)
    print(f'{len(plan)} combinations x {len(protocols)} protocols; '
          f'{len(done)} folds already on disk')

    aoa_rows = []
    started = time.perf_counter()
    completed = 0
    for target, regime, representation, model in plan:
        scoreable = models.scoreable_rows(mask, floored_mask, target)
        if representation == 'none':
            features, y = models.build_design(values[regime], mask, meta, target,
                                              elements, regime, 'raw', psi)
        else:
            features, y = models.build_design(values[regime], mask, meta, target,
                                              elements, regime, representation, psi)
        for protocol in protocols:
            rows = []
            for fold, train, test in protocol_splits(protocol, meta, block_id,
                                                     args.folds, args.repeats, SEED):
                key = (target, regime, representation, model, protocol, fold)
                if tuple(map(str, key)) in done:
                    continue
                predicted = models.fit_predict(model, train, test, features, y, meta,
                                               mask, bounds, target, xgb)
                keep = scoreable[test]
                rows.append({'target': target, 'regime': regime,
                             'representation': representation, 'model': model,
                             'protocol': protocol, 'fold': fold,
                             'n_train': int(train.sum()), 'n_test': int(test.sum()),
                             'n_scored': int(keep.sum()),
                             'r2': models.r_squared(y[test][keep],
                                                    np.asarray(predicted)[keep])})
                if (protocol == 'loso'
                        and (regime, representation, model) == PRIMARY):
                    inside, median_index = area_of_applicability(features, train, test)
                    aoa_rows.append({'target': target, 'fold': fold,
                                     'pct_inside_aoa': round(inside, 2),
                                     'median_dissimilarity': round(median_index, 4)})
            if rows:
                frame = pd.DataFrame(rows)
                frame.to_csv(checkpoint_path, mode='a', index=False,
                             header=not os.path.exists(checkpoint_path),
                             float_format='%.17g')
                completed += len(rows)
        elapsed = time.perf_counter() - started
        if completed:
            print(f'  {target:3} {regime:14} {representation:4} {model:18} '
                  f'{completed:6,} folds  {elapsed / 60:6.1f} min', flush=True)

    folds = pd.read_csv(checkpoint_path, float_precision='round_trip')
    if aoa_rows:
        pd.DataFrame(aoa_rows).to_csv(
            os.path.join(RESULTS, '07_area_of_applicability.csv'), index=False)

    summary = (folds.groupby(['target', 'regime', 'representation', 'model',
                              'protocol']).r2
               .agg(r2_median='median', r2_mean='mean', n_folds='size')
               .reset_index())
    wide = summary.pivot_table(index=['target', 'regime', 'representation', 'model'],
                               columns='protocol', values='r2_median').reset_index()
    if 'random' in wide.columns and 'loso' in wide.columns:
        wide['inflation_random_minus_loso'] = wide['random'] - wide['loso']
    if 'blocked' in wide.columns and 'loso' in wide.columns:
        wide['inflation_blocked_minus_loso'] = wide['blocked'] - wide['loso']
    wide.to_csv(os.path.join(RESULTS, '07_validation_summary.csv'), index=False)

    ratio = pd.read_csv(os.path.join(RESULTS, '01_lod_ratio.csv')) \
        if os.path.exists(os.path.join(RESULTS, '01_lod_ratio.csv')) else None
    lines = [
        '07_validate report',
        '',
        f'targets {len(targets)}   regimes {", ".join(regimes)}   '
        f'protocols {", ".join(protocols)}',
        f'folds on disk {len(folds):,}   wall clock this run '
        f'{(time.perf_counter() - started) / 60:.1f} min',
        '',
        'SPATIAL BLOCKS (variogram range, not a hand-chosen size)',
        '  ' + block_table.to_string(index=False).replace('\n', '\n  '),
        '',
        'MEDIAN R2 BY PROTOCOL, POOLED OVER TARGETS',
        '  ' + (wide.groupby(['regime', 'representation', 'model'])
               [[c for c in ('random', 'blocked', 'loso',
                             'inflation_random_minus_loso') if c in wide.columns]]
               .median().round(4).reset_index()
               .to_string(index=False).replace('\n', '\n  ')),
        '',
        '  Read the inflation column, not the random column. A random split and a',
        '  leave-one-survey-out split ask the same model the same question; the',
        '  difference between the answers is the part of the reported score that',
        '  was never generalisation.',
        '',
    ]
    if aoa_rows:
        aoa = pd.DataFrame(aoa_rows)
        lines += [
            'AREA OF APPLICABILITY, leave-one-survey-out, '
            f'{PRIMARY[0]}/{PRIMARY[1]}/{PRIMARY[2]}',
            '  ' + (aoa.groupby('fold')[['pct_inside_aoa', 'median_dissimilarity']]
                   .median().round(2).reset_index()
                   .to_string(index=False).replace('\n', '\n  ')),
            '',
            '  A held-out survey largely outside the area of applicability is being',
            '  extrapolated to, not predicted. That is a different failure from a',
            '  model that is simply wrong, and the two must not be reported as one.',
            '',
        ]
    lines += [
        'written to 05-results/: 07_validation_folds.csv, 07_validation_summary.csv,',
        '  07_spatial_blocks.csv, 07_area_of_applicability.csv, 07_validate_report.txt',
        '',
        'NOT DONE HERE',
        '  Bootstrap intervals, the inflation-versus-detection-limit-ratio',
        '  relationship, and the paired tests are 08_stats.py, which reads',
        '  07_validation_folds.csv. This script produces folds and nothing else,',
        '  so that a change to the statistics never requires refitting the models.',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '07_validate_report.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
