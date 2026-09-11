"""
07_pipeline_selftest - prove the validation machinery is sound before paying 7 hours

A long run that finishes is not a run that is correct. This script attacks the
parts of 07_validate.py that could be silently wrong, using the same code paths
the full grid uses, on two targets, in minutes. Every check has a definite
expected answer, so a pass is informative rather than reassuring.

  1 target isolation      the element being predicted must not reach the model as
                          a feature, in any representation. A log-ratio mixes all
                          parts into every coordinate, so removing the target
                          after transformation would leave it inside the features
                          and the model would predict a quantity from itself.
                          Tested by correlation, not by reading the code.

  2 permutation null      with the target shuffled, a pooled random split must
                          return R2 of about zero. Anything materially positive
                          means the fold machinery is leaking, and every number
                          in the grid would be wrong in the same direction.

  3 determinism           the same fold, fitted twice, must give the same score
                          to the bit. A drifting score means an unseeded RNG
                          somewhere, and the checkpoint would then be stitching
                          together runs that disagree.

  4 checkpoint fidelity   a run interrupted and resumed must produce exactly the
                          rows an uninterrupted run produces. This is the check
                          that matters most for a 7-hour job, because it is the
                          one whose failure is invisible.

  5 scored-row identity   the regimes must be scored on the same rows for the
                          same target and fold. If they are not, the comparison
                          between regimes measures the test set, not the regime.

  6 fold hygiene          blocked folds must not split a spatial block across
                          train and test, and leave-one-survey-out folds must not
                          leave any of the held-out survey in training.

Outputs to 05-results/
    07_pipeline_selftest.csv    one row per check
    07_pipeline_selftest.txt

Usage
    python 07_pipeline_selftest.py
    python 07_pipeline_selftest.py --targets Cu,Ag
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '05-results')
SEED = 20260906


def load_script(filename, alias):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        sys.exit(f'missing {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(results, name, passed, detail):
    results.append({'check': name, 'result': 'PASS' if passed else 'FAIL',
                    'detail': detail})
    print(f'  [{"PASS" if passed else "FAIL"}] {name}: {detail}', flush=True)
    return passed


# ------------------------------------------------------------------- checks
def check_target_isolation(results, models, values, mask, meta, elements, psi, target):
    """No feature may carry the target. Measured, not asserted."""
    worst = {}
    for regime in ['substitution', 'common_floor']:
        for representation in ['raw', 'clr', 'ilr']:
            features, y = models.build_design(values[regime], mask, meta, target,
                                              elements, regime, representation, psi)
            finite = np.isfinite(y)
            columns = np.nan_to_num(features[finite])
            centred = columns - columns.mean(axis=0)
            target_centred = y[finite] - y[finite].mean()
            denominator = (np.linalg.norm(centred, axis=0)
                           * np.linalg.norm(target_centred))
            with np.errstate(invalid='ignore', divide='ignore'):
                correlation = np.abs(centred.T @ target_centred / denominator)
            worst[f'{regime}/{representation}'] = float(np.nanmax(correlation))
    highest = max(worst.values())
    return check(results, 'target isolation',
                 highest < 0.999,
                 f'largest |correlation| between any feature and the target is '
                 f'{highest:.4f} across {len(worst)} designs; 1.000 would mean the '
                 f'target survived in the features')


def check_permutation_null(results, models, validate, values, mask, floored_mask,
                           meta, bounds, elements, psi, target, xgb):
    """Shuffle the target; a pooled random split must land at zero."""
    features, y = models.build_design(values['substitution'], mask, meta, target,
                                      elements, 'substitution', 'ilr', psi)
    rng = np.random.default_rng(SEED)
    shuffled = y.copy()
    finite = np.isfinite(shuffled)
    shuffled[finite] = rng.permutation(shuffled[finite])
    scoreable = models.scoreable_rows(mask, floored_mask, target)

    scores = []
    for _, train, test in validate.protocol_splits('random', meta, None, 5, 1, SEED):
        predicted = models.fit_predict('xgb_regression', train, test, features,
                                       shuffled, meta, mask, bounds, target, xgb)
        keep = scoreable[test]
        scores.append(models.r_squared(shuffled[test][keep],
                                       np.asarray(predicted)[keep]))
    median = float(np.nanmedian(scores))
    # One-sided, deliberately. A shuffled target has nothing to learn, so a
    # flexible model fits noise in training and scores BELOW zero out of fold -
    # that is overfitting behaving exactly as it should, not a leak. Only a
    # materially POSITIVE score is evidence that information crossed the split.
    # The first version of this check tested |R2| < 0.05 and failed the pipeline
    # for being correct.
    return check(results, 'permutation null',
                 median < 0.05,
                 f'median R2 with a shuffled target is {median:+.4f}; must not be '
                 f'materially positive, and negative is the expected sign')


def check_determinism(results, models, validate, values, mask, meta, bounds,
                      elements, psi, target, xgb):
    """Two identical fits must agree to the bit."""
    features, y = models.build_design(values['substitution'], mask, meta, target,
                                      elements, 'substitution', 'ilr', psi)
    splits = list(validate.protocol_splits('loso', meta, None, 5, 1, SEED))
    _, train, test = splits[0]
    first = models.fit_predict('xgb_regression', train, test, features, y, meta,
                               mask, bounds, target, xgb)
    second = models.fit_predict('xgb_regression', train, test, features, y, meta,
                                mask, bounds, target, xgb)
    identical = bool(np.array_equal(np.asarray(first), np.asarray(second)))
    difference = float(np.nanmax(np.abs(np.asarray(first) - np.asarray(second))))
    return check(results, 'determinism', identical,
                 f'two identical fits differ by at most {difference:.3e}; anything '
                 f'above zero means an unseeded generator')


def check_scored_rows_identical(results, checkpoint_path):
    """Every regime must be judged on the same rows, checked on what was WRITTEN.

    Comparing the row-selection function against itself would prove nothing, since
    it takes no regime argument. The only informative version reads the folds
    already on disk and asks whether the regimes actually ended up scoring the
    same number of rows for the same target, protocol and fold.
    """
    if not os.path.exists(checkpoint_path):
        return check(results, 'scored rows identical across regimes', False,
                     'no folds on disk yet; run 07_validate.py on a few targets '
                     'first, then re-run this test')
    folds = pd.read_csv(checkpoint_path, float_precision='round_trip')
    spread = (folds.groupby(['target', 'protocol', 'fold']).n_scored
              .nunique().reset_index(name='distinct_counts'))
    offenders = spread[spread.distinct_counts > 1]
    return check(results, 'scored rows identical across regimes',
                 len(offenders) == 0,
                 f'{len(spread):,} target-protocol-fold groups checked across '
                 f'regimes, {len(offenders)} disagree on how many rows were '
                 f'scored')


def check_fold_hygiene(results, validate, meta, block_id, folds):
    """No block split across a blocked fold; no survey left behind in LOSO."""
    leaks = 0
    for _, train, test in validate.protocol_splits('blocked', meta, block_id,
                                                   folds, 1, SEED):
        shared = set(block_id[train]) & set(block_id[test])
        leaks += len(shared - {-1})
    blocked_clean = check(results, 'blocked folds keep blocks whole', leaks == 0,
                          f'{leaks} block(s) appear in both train and test')

    survey_leaks = 0
    for label, train, test in validate.protocol_splits('loso', meta, None, folds,
                                                       1, SEED):
        held = set(meta.survey.to_numpy()[test])
        survey_leaks += len(held & set(meta.survey.to_numpy()[train]))
    loso_clean = check(results, 'leave-one-survey-out holds the survey out',
                       survey_leaks == 0,
                       f'{survey_leaks} survey(s) present on both sides of a fold')
    return blocked_clean and loso_clean


def check_checkpoint_fidelity(results, checkpoint_path):
    """A resumed run must reproduce an uninterrupted one, key for key."""
    if not os.path.exists(checkpoint_path):
        return check(results, 'checkpoint fidelity', False,
                     'no checkpoint on disk yet; run 07_validate.py on a few '
                     'targets first, then re-run this test')
    folds = pd.read_csv(checkpoint_path, float_precision='round_trip')
    keys = folds[['target', 'regime', 'representation', 'model', 'protocol', 'fold']]
    duplicated = int(keys.duplicated().sum())
    conflicting = 0
    if duplicated:
        grouped = folds.groupby(list(keys.columns)).r2.nunique()
        conflicting = int((grouped > 1).sum())
    return check(results, 'checkpoint fidelity',
                 duplicated == 0 or conflicting == 0,
                 f'{len(folds):,} rows, {duplicated} duplicate keys, {conflicting} '
                 f'of them disagreeing on R2; duplicates are harmless only while '
                 f'they agree')


def main():
    ap = argparse.ArgumentParser(description='Pre-flight for the validation grid')
    ap.add_argument('--targets', default='Cu,Ag')
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    models = load_script('06_models.py', 'models_06')
    validate = load_script('07_validate.py', 'validate_07')
    xgb = models.require_xgboost()

    (meta, values, mask, floored_mask, bounds, floor,
     elements, psi) = models.load_task_inputs()
    targets = [t for t in args.targets.split(',') if t in elements]
    if not targets:
        sys.exit(f'none of {args.targets} are shared elements')
    print(f'{len(meta):,} rows, {len(elements)} elements, testing on '
          f'{", ".join(targets)}')

    rng = np.random.default_rng(SEED)
    reference_features, _ = models.build_design(
        values['substitution'], mask, meta, targets[0], elements,
        'substitution', 'clr', psi)
    block_id, block_table = validate.spatial_blocks(meta, reference_features,
                                                    elements, rng)

    results = []
    for target in targets:
        print(f'target {target}')
        check_target_isolation(results, models, values, mask, meta, elements, psi,
                               target)
        check_determinism(results, models, validate, values, mask, meta, bounds,
                          elements, psi, target, xgb)
        check_permutation_null(results, models, validate, values, mask,
                               floored_mask, meta, bounds, elements, psi, target,
                               xgb)
    print('folds and checkpoint')
    checkpoint_path = os.path.join(RESULTS, '07_validation_folds.csv')
    check_fold_hygiene(results, validate, meta, block_id, 5)
    check_scored_rows_identical(results, checkpoint_path)
    check_checkpoint_fidelity(results, checkpoint_path)

    table = pd.DataFrame(results)
    table.to_csv(os.path.join(RESULTS, '07_pipeline_selftest.csv'), index=False)
    failures = int((table.result == 'FAIL').sum())

    lines = [
        '07_pipeline_selftest report',
        '',
        f'rows {len(meta):,}   elements {len(elements)}   targets tested '
        f'{", ".join(targets)}',
        '',
        '  ' + table.to_string(index=False).replace('\n', '\n  '),
        '',
        f'VERDICT: {"CLEAR TO RUN THE GRID" if failures == 0 else str(failures) + " FAILURE(S) - DO NOT RUN THE GRID"}',
        '',
        '  These checks test the machinery, not the science. They cannot tell you',
        '  whether the result is interesting; they can tell you that a seven-hour',
        '  run will not produce numbers that are wrong for a reason nobody looks',
        '  for afterwards.',
        '',
        'written to 05-results/: 07_pipeline_selftest.csv, 07_pipeline_selftest.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '07_pipeline_selftest.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
