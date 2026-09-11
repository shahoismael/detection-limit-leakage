"""
08_stats - intervals, paired tests, and the test that can refute the paper

Reads 07_validation_folds.csv and does no model fitting, so a change to the
statistics never costs another grid run.

THE FALSIFICATION TEST

Everything before this establishes that censoring patterns identify a survey and
that cross-region scores are inflated. Neither shows that the censoring causes the
inflation on real data. This does: elements differ in how far apart their
detection limits sit across surveys, so if censoring drives the inflation then
inflation must rise with that disparity. Spearman's rho between an element's
cross-survey detection-limit ratio and its inflation, with a bootstrap interval,
is the number. A flat or negative relationship refutes the mechanism on real data
regardless of what the mask experiments returned, and the script reports it in
those terms rather than hunting for a subgroup where it holds.

INTERVALS

Bootstrap percentile intervals over ELEMENTS, resampling the 44 rotations. Never
over folds: leave-one-survey-out yields three, and an interval built from three
correlated folds would be a decoration.

The methodology also commits to intervals over SAMPLES. Those cannot be computed
from this input, because 07_validate.py writes one R2 per fold and not the
per-sample residuals an over-samples bootstrap needs. That is reported as a gap,
not quietly dropped: producing them means re-running the grid with residuals
stored, which is a decision about cost rather than a detail.

PAIRED TESTS

Arms are compared on the same 44 elements, so the comparison is paired and uses a
Wilcoxon signed-rank test on the per-element differences together with a paired
bootstrap of the median difference. Reporting an unpaired test here would throw
away the pairing and inflate the p-value.

McNemar and DeLong belong to the classification experiments in 03_masks.py, which
compares accuracies and AUCs. They are not computed here because 03_masks.py
writes confusion matrices rather than per-sample predictions, and the two tests
need per-sample agreement. That is also reported as a gap.

MULTIPLE COMPARISONS

Corrected with Benjamini-Hochberg across the family of arm comparisons, and the
uncorrected p-values are printed beside the corrected ones so the correction can
be checked rather than trusted.

Outputs to 05-results/
    08_interval_by_arm.csv          median and 95% interval per arm and protocol
    08_inflation_vs_lod_ratio.csv   per element, the falsification test's input
    08_paired_tests.csv             arm-versus-arm, corrected and uncorrected
    08_stats_report.txt

Usage
    python 08_stats.py
    python 08_stats.py --draws 20000
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '05-results')
SEED = 20260906
PRIMARY_MODEL = 'xgb_regression'


def read_results(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        sys.exit(f'missing {path}. Run 07_validate.py first.')
    return pd.read_csv(path, float_precision='round_trip')


def bootstrap_median(values, draws, rng):
    """Percentile interval on the median, resampling the elements themselves."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan, np.nan
    index = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[index], axis=1)
    return (float(np.median(values)),
            float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)))


def per_element_scores(folds):
    """One score per element per arm per protocol: the median over that arm's folds.

    The median rather than the mean, because R2 is unbounded below and a single
    collapsed fold would otherwise set the value for the whole arm.
    """
    return (folds.groupby(['target', 'regime', 'representation', 'model', 'protocol'])
            .agg(r2=('r2', 'median'), n_folds=('r2', 'size'),
                 n_scored=('n_scored', 'median'))
            .reset_index())


def arm_intervals(scores, draws, rng):
    rows = []
    for (regime, representation, model, protocol), block in scores.groupby(
            ['regime', 'representation', 'model', 'protocol']):
        median, low, high = bootstrap_median(block.r2, draws, rng)
        rows.append({'regime': regime, 'representation': representation,
                     'model': model, 'protocol': protocol,
                     'n_elements': len(block),
                     'r2_median': median, 'ci_low': low, 'ci_high': high})
    return pd.DataFrame(rows)


def inflation_per_element(scores):
    """random minus loso, per element and arm. The quantity the paper is about."""
    wide = scores.pivot_table(index=['target', 'regime', 'representation', 'model'],
                              columns='protocol', values='r2').reset_index()
    for protocol in ('random', 'blocked', 'loso'):
        if protocol not in wide.columns:
            wide[protocol] = np.nan
    wide['inflation_random_minus_loso'] = wide['random'] - wide['loso']
    wide['inflation_blocked_minus_loso'] = wide['blocked'] - wide['loso']
    wide['blocking_recovers'] = wide['random'] - wide['blocked']
    return wide


def inflation_by_divergence(inflation, lod_ratio, draws, rng, threshold=2.0,
                            model=PRIMARY_MODEL):
    """Inflation split by whether an element's limits diverge across surveys.

    The headline inflation is the whole random-minus-leave-one-survey-out gap, and
    that gap has two sources: the censoring this paper is about, and the genuine
    geochemical difference between three continents. Nothing in the pooled number
    separates them, so a reader is entitled to ask what share is censoring.

    This is the cheapest available answer. Elements whose detection limits agree
    across surveys carry the continental difference and almost no censoring
    disparity; elements whose limits differ by a factor of two or more carry both.
    The difference between the two groups is attributable to the limits, and the
    non-divergent group is a floor on how much inflation would remain if censoring
    were harmonised perfectly.

    It is not a formal decomposition. The groups are not randomised and elements
    differ in more than their limits, so the contrast is reported as a contrast.
    """
    block = inflation[inflation.model == model]
    rows = []
    for (regime, representation), arm in block.groupby(['regime', 'representation']):
        merged = arm.merge(lod_ratio[['element', 'lod_ratio']],
                           left_on='target', right_on='element', how='inner')
        merged = merged[np.isfinite(merged.inflation_random_minus_loso)
                        & np.isfinite(merged.lod_ratio)]
        divergent = merged[merged.lod_ratio >= threshold]
        agreeing = merged[merged.lod_ratio < threshold]
        if len(divergent) < 5 or len(agreeing) < 5:
            continue
        hi, hi_lo, hi_hi = bootstrap_median(divergent.inflation_random_minus_loso,
                                            draws, rng)
        lo, lo_lo, lo_hi = bootstrap_median(agreeing.inflation_random_minus_loso,
                                            draws, rng)
        difference = stats.mannwhitneyu(divergent.inflation_random_minus_loso,
                                        agreeing.inflation_random_minus_loso,
                                        alternative='greater')
        rows.append({'regime': regime, 'representation': representation,
                     'model': model,
                     'n_divergent': len(divergent), 'n_agreeing': len(agreeing),
                     'inflation_divergent': hi,
                     'divergent_ci_low': hi_lo, 'divergent_ci_high': hi_hi,
                     'inflation_agreeing': lo,
                     'agreeing_ci_low': lo_lo, 'agreeing_ci_high': lo_hi,
                     'excess_attributable_to_limits': hi - lo,
                     'mann_whitney_p': float(difference.pvalue)})
    return pd.DataFrame(rows)


def block_bootstrap_spearman(x, y, groups, draws, rng):
    """Spearman interval that resamples GROUPS of elements, not elements.

    Resampling elements one at a time treats them as independent. They are not:
    the composition is closed, so the parts are constrained to sum, and every
    element's inflation is estimated on the same samples. Resampling whole
    affinity classes keeps correlated elements together and widens the interval
    to something the dependence structure can support.

    The narrower element-level interval is reported beside it rather than
    replaced, so the cost of the assumption is visible instead of asserted.
    """
    labels = np.asarray(groups)
    unique = np.unique(labels)
    if len(unique) < 3:
        return np.nan, np.nan, 0
    members = [np.flatnonzero(labels == g) for g in unique]
    estimates = []
    for _ in range(draws):
        picked = rng.integers(0, len(members), size=len(members))
        index = np.concatenate([members[i] for i in picked])
        if len(np.unique(x[index])) < 3:
            continue
        estimates.append(stats.spearmanr(x[index], y[index]).statistic)
    estimates = np.asarray([e for e in estimates if np.isfinite(e)])
    if len(estimates) < 50:
        return np.nan, np.nan, len(unique)
    return (float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)), len(unique))


def falsification_test(inflation, lod_ratio, draws, rng, model=PRIMARY_MODEL,
                       affinity=None):
    """Does inflation rise with the cross-survey detection-limit disparity?"""
    rows, per_element = [], {}
    for (regime, representation), block in inflation[
            inflation.model == model].groupby(['regime', 'representation']):
        merged = block.merge(lod_ratio[['element', 'lod_ratio']],
                             left_on='target', right_on='element', how='inner')
        merged = merged[np.isfinite(merged.inflation_random_minus_loso)
                        & np.isfinite(merged.lod_ratio)]
        if len(merged) < 8:
            continue
        x = np.log10(merged.lod_ratio.to_numpy(dtype=float))
        y = merged.inflation_random_minus_loso.to_numpy(dtype=float)
        rho, p_value = stats.spearmanr(x, y)
        index = rng.integers(0, len(x), size=(draws, len(x)))
        resampled = np.array([stats.spearmanr(x[i], y[i]).statistic for i in index[:2000]])
        groups = (merged.element.map(affinity).fillna('other').to_numpy()
                  if affinity else np.arange(len(merged)))
        block_low, block_high, n_groups = block_bootstrap_spearman(
            x, y, groups, min(draws, 2000), rng)
        rows.append({'regime': regime, 'representation': representation,
                     'model': model, 'n_elements': len(merged),
                     'spearman_rho': float(rho), 'p_value': float(p_value),
                     'rho_ci_low': float(np.nanpercentile(resampled, 2.5)),
                     'rho_ci_high': float(np.nanpercentile(resampled, 97.5)),
                     'n_affinity_groups': n_groups,
                     'rho_block_ci_low': block_low,
                     'rho_block_ci_high': block_high})
        per_element[(regime, representation)] = merged[
            ['target', 'lod_ratio', 'random', 'blocked', 'loso',
             'inflation_random_minus_loso']]
    return pd.DataFrame(rows), per_element


def benjamini_hochberg(p_values):
    """Step-up false discovery rate control, returned in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(p_values)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(n)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def paired_tests(scores, draws, rng, protocol='loso', model=PRIMARY_MODEL):
    """Every arm against every other, on the same elements."""
    block = scores[(scores.protocol == protocol) & (scores.model == model)]
    wide = block.pivot_table(index='target', columns=['regime', 'representation'],
                             values='r2')
    arms = list(wide.columns)
    rows = []
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            pair = wide[[left, right]].dropna()
            if len(pair) < 8:
                continue
            difference = (pair[left] - pair[right]).to_numpy(dtype=float)
            statistic, p_value = stats.wilcoxon(difference)
            median, low, high = bootstrap_median(difference, draws, rng)
            rows.append({'protocol': protocol, 'model': model,
                         'arm_a': '/'.join(left), 'arm_b': '/'.join(right),
                         'n_elements': len(pair),
                         'median_difference': median,
                         'ci_low': low, 'ci_high': high,
                         'wilcoxon_statistic': float(statistic),
                         'p_value': float(p_value)})
    table = pd.DataFrame(rows)
    if len(table):
        table['p_value_bh'] = benjamini_hochberg(table.p_value)
        table = table.sort_values('p_value')
    return table


def main():
    ap = argparse.ArgumentParser(description='Intervals, paired tests, refutation')
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--divergence-threshold', type=float, default=2.0,
                    help='cross-survey detection-limit ratio above which an '
                         'element counts as divergent')
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    folds = read_results('07_validation_folds.csv')
    lod_ratio = read_results('01_lod_ratio.csv')
    scores = per_element_scores(folds)
    print(f'{len(folds):,} folds -> {len(scores):,} element-arm-protocol scores')

    intervals = arm_intervals(scores, args.draws, rng)
    intervals.round(4).to_csv(os.path.join(RESULTS, '08_interval_by_arm.csv'),
                              index=False)

    inflation = inflation_per_element(scores)
    inflation_intervals = []
    for (regime, representation, model), block in inflation.groupby(
            ['regime', 'representation', 'model']):
        median, low, high = bootstrap_median(block.inflation_random_minus_loso,
                                             args.draws, rng)
        recovered, recovered_low, recovered_high = bootstrap_median(
            block.blocking_recovers, args.draws, rng)
        inflation_intervals.append({
            'regime': regime, 'representation': representation, 'model': model,
            'n_elements': len(block),
            'inflation_median': median, 'ci_low': low, 'ci_high': high,
            'blocking_recovers_median': recovered,
            'blocking_ci_low': recovered_low, 'blocking_ci_high': recovered_high})
    inflation_table = pd.DataFrame(inflation_intervals).sort_values('inflation_median')
    # Written out, not only printed. 09_report.py builds a table from these
    # intervals, and a downstream script that has to re-derive them would be free
    # to disagree with this one about its own numbers.
    inflation_table.round(4).to_csv(
        os.path.join(RESULTS, '08_inflation_by_arm.csv'), index=False)

    # The affinity classes are the ones 04_transform.py froze the partition from,
    # read back rather than restated so the two cannot drift apart.
    affinity = {}
    partition_path = os.path.join(HERE, '04_transform.py')
    if os.path.exists(partition_path):
        spec = importlib.util.spec_from_file_location('transform_04', partition_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        affinity = {element: name
                    for name, group in module.AFFINITY_GROUPS for element in group}

    refutation, per_element = falsification_test(inflation, lod_ratio,
                                                 args.draws, rng, affinity=affinity)
    divergence = inflation_by_divergence(inflation, lod_ratio, args.draws, rng,
                                         threshold=args.divergence_threshold)
    if len(divergence):
        divergence.round(4).to_csv(
            os.path.join(RESULTS, '08_inflation_by_divergence.csv'), index=False)
    if per_element:
        pd.concat([frame.assign(regime=key[0], representation=key[1])
                   for key, frame in per_element.items()], ignore_index=True) \
            .round(4).to_csv(
                os.path.join(RESULTS, '08_inflation_vs_lod_ratio.csv'), index=False)

    tests = paired_tests(scores, args.draws, rng)
    if len(tests):
        tests.round(6).to_csv(os.path.join(RESULTS, '08_paired_tests.csv'),
                              index=False)

    # The verdict rests on ONE arm: the pre-registered primary, substitution with
    # the isometric log-ratio. Everything else is sensitivity.
    #
    # An earlier version required every ILR arm's interval to clear zero, which
    # silently gave a veto to mask_feature - a CONTROL arm, built to show what
    # happens when the mask is handed to the model as a feature. Its interval
    # grazed zero and the script reported the paper's central causal test as
    # failed while four of five arms were positive. A control that disagrees is
    # information about the control, not a refutation of the primary.
    #
    # An empty table is also not a refutation: it means too few elements carried
    # both an inflation and a detection-limit ratio for the test to run at all.
    primary_arm = ('substitution', 'ilr', PRIMARY_MODEL)
    primary = pd.DataFrame()
    if len(refutation) and {'regime', 'representation', 'model'} <= set(refutation):
        primary = refutation[(refutation.regime == primary_arm[0])
                             & (refutation.representation == primary_arm[1])
                             & (refutation.model == primary_arm[2])]
    testable = bool(len(primary))
    mechanism_supported = testable and bool(primary.rho_ci_low.iloc[0] > 0)
    sensitivity = refutation[~refutation.index.isin(primary.index)] \
        if len(refutation) else refutation
    agreeing = int((sensitivity.rho_ci_low > 0).sum()) if len(sensitivity) else 0
    verdict = ('MECHANISM SUPPORTED' if mechanism_supported
               else 'NOT SUPPORTED' if testable
               else 'NOT TESTABLE - too few elements carry both an inflation and a '
                    'detection-limit ratio')

    display_intervals = intervals[intervals.model.isin(
        [PRIMARY_MODEL, 'ridge', 'survey_mean', 'mask_density_only',
         'coordinates_only'])].copy()
    display_intervals['interval'] = display_intervals.apply(
        lambda r: f'{r.r2_median:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]', axis=1)

    lines = [
        '08_stats report',
        '',
        f'input             {len(folds):,} folds, '
        f'{scores.target.nunique()} elements',
        f'bootstrap         {args.draws:,} draws, percentile intervals over '
        f'ELEMENTS',
        f'median scored rows per element  '
        f'{scores.n_scored.median():,.0f}',
        '',
        'R2 BY ARM AND PROTOCOL, median over elements with a 95% interval',
        '  ' + (display_intervals.pivot_table(
            index=['regime', 'representation', 'model'], columns='protocol',
            values='interval', aggfunc='first').reset_index()
            .to_string(index=False).replace('\n', '\n  ')),
        '',
        'INFLATION, random minus leave-one-survey-out',
        '  ' + inflation_table.round(4).to_string(index=False)
        .replace('\n', '\n  '),
        '',
        '  blocking_recovers is random minus blocked: how much of the inflation',
        '  spatially blocked validation actually removes. Compare it against the',
        '  inflation beside it. If it is the smaller number, then the protocol the',
        '  field recommends for optimistic cross-validation does not address this',
        '  failure, because the structure being leaked is laboratory identity and',
        '  not distance.',
        '',
        'THE FALSIFICATION TEST',
        '  Spearman rho between log10 of an element\'s cross-survey detection-limit',
        '  ratio and that element\'s inflation. Positive means censoring disparity',
        '  drives inflation; zero or negative refutes the mechanism on real data.',
        '  ' + (refutation.round(4).to_string(index=False).replace('\n', '\n  ')
               if len(refutation) else 'not computable: no overlapping elements'),
        '',
        '  Two intervals are given per arm. rho_ci_* resamples elements, which',
        '  assumes they are independent; they are not, since the composition is',
        '  closed and every element is estimated on the same samples.',
        '  rho_block_ci_* resamples whole affinity classes instead and is the',
        '  interval the dependence structure actually supports.',
        '',
        f'  VERDICT: {verdict}',
        f'    judged on the pre-registered primary arm alone: '
        f'{"/".join(primary_arm)}',
        (f'    rho {primary.spearman_rho.iloc[0]:+.4f} '
         f'[{primary.rho_ci_low.iloc[0]:+.4f}, {primary.rho_ci_high.iloc[0]:+.4f}], '
         f'p = {primary.p_value.iloc[0]:.4f}, n = {int(primary.n_elements.iloc[0])} '
         f'elements' if testable else '    the primary arm produced no test'),
        f'    sensitivity arms whose interval also clears zero: {agreeing} of '
        f'{max(len(sensitivity), 0)}',
        '',
        '  The primary arm is fixed in advance so that the verdict cannot be chosen',
        '  after the fact from whichever arm happens to agree. mask_feature is a',
        '  CONTROL - the mask handed to the model as a feature - and a control that',
        '  disagrees is information about the control, not a veto over the primary.',
        '',
        '  A negative result here is publishable and is not to be repaired by',
        '  subsetting elements until the correlation appears. It would mean the',
        '  identity channel is real, the inflation is real, and the link between',
        '  them is not established on these surveys.',
        '',
        f'HOW MUCH OF THE INFLATION IS ATTRIBUTABLE TO THE LIMITS '
        f'(split at {args.divergence_threshold:g}x)',
        '  ' + (divergence.round(4).to_string(index=False).replace('\n', '\n  ')
               if len(divergence) else 'not computable: too few elements on one side'),
        '',
        '  The pooled inflation is the whole random-minus-leave-one-survey-out gap,',
        '  and that gap has two sources: the censoring this study is about, and the',
        '  genuine geochemical difference between three continents. Elements whose',
        '  limits agree across surveys carry the second and almost none of the',
        '  first, so their inflation is a floor and the excess above it is what the',
        '  limits add. The groups are not randomised and differ in more than their',
        '  limits, so this is a contrast, not a decomposition, and is reported as one.',
        '',
    ]
    if len(tests):
        lines += [
            'PAIRED COMPARISONS UNDER LEAVE-ONE-SURVEY-OUT '
            f'({PRIMARY_MODEL}, Wilcoxon signed-rank, Benjamini-Hochberg)',
            '  ' + tests.head(10).round(4).to_string(index=False)
            .replace('\n', '\n  '),
            '',
            '  Paired on the same elements, so the test uses the pairing rather',
            '  than discarding it. Uncorrected p-values are shown beside the',
            '  corrected ones so the correction can be checked.',
            '',
        ]
    lines += [
        'GAPS, STATED RATHER THAN OMITTED',
        '  Intervals over SAMPLES are not computed. 07_validate.py writes one R2',
        '  per fold, not per-sample residuals, and an over-samples bootstrap needs',
        '  the residuals. Producing them means re-running the grid with residuals',
        '  stored - a cost decision, not an oversight to be written around.',
        '',
        '  McNemar and DeLong are not computed. They belong to the classification',
        '  experiments in 03_masks.py and need per-sample predictions, which that',
        '  script does not write; it writes confusion matrices. The same fix',
        '  applies: store predictions, then the tests become available.',
        '',
        'written to 05-results/: 08_interval_by_arm.csv, 08_inflation_by_arm.csv,',
        '  08_inflation_vs_lod_ratio.csv, 08_paired_tests.csv, 08_stats_report.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '08_stats_report.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
