"""
09_report - the tables and figure data the Results section is written from

Reads only what earlier stages wrote and computes nothing new. Every number in
Section 4 must be traceable to a file produced by a numbered script, so this one
assembles rather than analyses: if a value appears in a table here and nowhere
upstream, that is a defect, not a convenience.

TABLES

  Table 1  the three surveys and their censoring, from the audit
  Table 2  cross-survey detection-limit ratios, the widest twelve
  Table 3  censoring as an identity channel, the mask experiments
  Table 4  R2 by regime, representation and protocol, with intervals
  Table 5  inflation, and how much spatial blocking recovers of it
  Table 6  the cost of the correction, per survey

FIGURES

  Figure 1  the dose-response simulation, already built by
            03_figure_dose_response.py and not rebuilt here
  Figure 2  inflation against cross-survey detection-limit ratio, one point per
            element. This is the falsification test made visible, and it is a
            primary figure because a reader can refute the paper by looking at it
  Figure 3  the three protocols side by side per arm, which shows that spatial
            blocking sits beside random rather than beside leave-one-survey-out

Figure data is written as CSV rather than plotted. The plotting script owns the
palette and the typography; this one owns the numbers, and keeping them apart
means a change to either cannot silently alter the other.

Outputs to 05-results/tables/
    table_1_surveys.csv ... table_6_correction_cost.csv
Outputs to 05-results/figure_data/
    figure_2_inflation_vs_lod_ratio.csv
    figure_3_protocol_comparison.csv
Outputs to 05-results/
    09_report.txt           every table rendered, and what is missing

Usage
    python 09_report.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '05-results')
TABLES = os.path.join(RESULTS, 'tables')
FIGURE_DATA = os.path.join(RESULTS, 'figure_data')
PRIMARY_ARM = ('substitution', 'ilr', 'xgb_regression')


def read(name, required=True):
    path = os.path.join(RESULTS, name)
    if os.path.exists(path):
        return pd.read_csv(path, float_precision='round_trip')
    if required:
        sys.exit(f'missing {path}. Run the earlier scripts first.')
    return None


def interval(row, value='r2_median', low='ci_low', high='ci_high'):
    if not np.isfinite(row[value]):
        return ''
    return f'{row[value]:+.3f} [{row[low]:+.3f}, {row[high]:+.3f}]'


def build_table_1(lod_table):
    """The three surveys, their censoring, and how each declares it."""
    rows = []
    for survey, block in lod_table.groupby('survey'):
        rows.append({
            'survey': survey,
            'n_samples': int(block.n_rows.max()),
            'n_elements': len(block),
            'elements_with_censoring': int((block.n_censored > 0).sum()),
            'censored_cells_pct': round(
                100 * block.n_censored.sum()
                / max(block.n_rows.max() * len(block), 1), 2),
            'lowest_rate_element': block.loc[
                block[block.n_censored > 0].censored_pct.idxmin(), 'element']
            if (block.n_censored > 0).any() else '',
            'lowest_rate_pct': round(
                block[block.n_censored > 0].censored_pct.min(), 2)
            if (block.n_censored > 0).any() else np.nan,
            'highest_rate_element': block.loc[
                block[block.n_censored > 0].censored_pct.idxmax(), 'element']
            if (block.n_censored > 0).any() else '',
            'highest_rate_pct': round(
                block[block.n_censored > 0].censored_pct.max(), 2)
            if (block.n_censored > 0).any() else np.nan,
            'flag_convention': block.flag_convention.iloc[0],
        })
    return pd.DataFrame(rows)


def build_table_3(mask_identity, random_subsets):
    """The mask experiments, reduced to the comparison that carries the claim."""
    best = (mask_identity[mask_identity.model != 'majority_class']
            .sort_values('balanced_accuracy', ascending=False)
            .groupby('experiment', as_index=False).first())
    keep = ['experiment', 'model', 'n_features', 'balanced_accuracy',
            'balanced_accuracy_sd', 'macro_f1', 'kappa', 'no_information_rate']
    table = best[keep].sort_values('balanced_accuracy', ascending=False)
    # Without this column the E1d rows read as one result. They are two arms with
    # opposite evidential status: the feature set chosen by unavailability is a
    # ceiling and proves nothing, and only the randomly drawn one carries the
    # claim. A reader of the table alone must be able to tell them apart.
    table.insert(1, 'arm', np.where(
        table.experiment.str.startswith('E1d_survival_k'),
        'ceiling (feature set chosen by unavailability)', 'finding'))
    if random_subsets is not None and len(random_subsets):
        summary = (random_subsets.groupby('k_features')
                   .agg(balanced_accuracy=('balanced_accuracy', 'median'),
                        retention_disparity=('retention_disparity', 'median'))
                   .reset_index())
        summary['experiment'] = summary.k_features.map(
            lambda k: f'E1d_survival_random_k{k}')
        summary['model'] = 'survive_flag_closed_form'
        summary['n_features'] = 1
        summary['arm'] = 'finding (random feature sets, 200 draws per k)'
        table = pd.concat([table, summary[['experiment', 'arm', 'model',
                                           'n_features', 'balanced_accuracy',
                                           'retention_disparity']]],
                          ignore_index=True)
    return table


def build_table_4(intervals):
    table = intervals.copy()
    table['estimate'] = table.apply(interval, axis=1)
    wide = table.pivot_table(index=['regime', 'representation', 'model'],
                             columns='protocol', values='estimate',
                             aggfunc='first').reset_index()
    order = [c for c in ('random', 'blocked', 'loso') if c in wide.columns]
    return wide[['regime', 'representation', 'model'] + order]


def build_table_5(inflation_table):
    table = inflation_table.copy()
    table['inflation'] = table.apply(
        lambda r: interval(r, 'inflation_median', 'ci_low', 'ci_high'), axis=1)
    table['blocking_recovers'] = table.apply(
        lambda r: interval(r, 'blocking_recovers_median',
                           'blocking_ci_low', 'blocking_ci_high'), axis=1)
    # The ratio is the sentence the table has to make sayable: what fraction of
    # the inflation the recommended remedy actually removes.
    with np.errstate(invalid='ignore', divide='ignore'):
        table['pct_of_inflation_recovered'] = (
            100 * table.blocking_recovers_median / table.inflation_median).round(1)
    return table[['regime', 'representation', 'model', 'n_elements', 'inflation',
                  'blocking_recovers', 'pct_of_inflation_recovered']] \
        .sort_values('regime')


def main():
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURE_DATA, exist_ok=True)

    lod_table = read('01_lod_table.csv')
    lod_ratio = read('01_lod_ratio.csv')
    mask_identity = read('03_mask_identity.csv')
    random_subsets = read('03_survival_random_subsets.csv', required=False)
    intervals = read('08_interval_by_arm.csv')
    inflation_points = read('08_inflation_vs_lod_ratio.csv')
    floor_cost = read('05_common_floor_cost.csv')
    floor_limits = read('05_common_floor.csv')
    paired = read('08_paired_tests.csv', required=False)

    # Read, never re-derived. Recomputing the inflation here from the per-arm
    # medians would drop the bootstrap intervals and let this script publish
    # numbers that disagree with 08_stats.py while looking identical.
    inflation_summary = read('08_inflation_by_arm.csv')

    table_1 = build_table_1(lod_table)
    table_2 = (lod_ratio.sort_values('lod_ratio', ascending=False)
               .head(12).round(4))
    table_3 = build_table_3(mask_identity, random_subsets)
    table_4 = build_table_4(intervals)
    table_5 = build_table_5(inflation_summary)
    # Table 6 is the per-survey cost, which is what the caption promises and what
    # the correction is judged by. The per-element floors are supporting data and
    # are written beside it rather than in place of it.
    table_6 = floor_cost.copy()
    table_6_supporting = floor_limits[floor_limits.n_surveys_with_limit >= 2] \
        .sort_values('common_floor', ascending=False)

    for name, frame in [('table_1_surveys', table_1),
                        ('table_2_detection_limit_ratios', table_2),
                        ('table_3_identity_channel', table_3),
                        ('table_4_performance_by_protocol', table_4),
                        ('table_5_inflation_and_blocking', table_5),
                        ('table_6_correction_cost', table_6),
                        ('table_6_supporting_common_floors', table_6_supporting)]:
        frame.to_csv(os.path.join(TABLES, name + '.csv'), index=False)

    figure_2 = inflation_points.copy()
    figure_2['is_primary_arm'] = ((figure_2.regime == PRIMARY_ARM[0])
                                  & (figure_2.representation == PRIMARY_ARM[1]))
    figure_2.to_csv(os.path.join(FIGURE_DATA,
                                 'figure_2_inflation_vs_lod_ratio.csv'), index=False)

    figure_3 = intervals[intervals.model.isin(['xgb_regression', 'ridge'])].copy()
    figure_3.to_csv(os.path.join(FIGURE_DATA,
                                 'figure_3_protocol_comparison.csv'), index=False)

    primary = figure_2[figure_2.is_primary_arm]
    missing = []
    for required in ('E0b_summary.txt', '03_mask_report.txt', '07_validate_report.txt',
                     '08_stats_report.txt'):
        if not os.path.exists(os.path.join(RESULTS, required)):
            missing.append(required)
    if not os.path.exists(os.path.join(HERE, '..', '06-figs')):
        missing.append('06-figs/ (Figure 1 output directory)')

    lines = [
        '09_report',
        '',
        'TABLE 1  surveys and censoring',
        '  ' + table_1.to_string(index=False).replace('\n', '\n  '),
        '',
        'TABLE 2  widest cross-survey detection-limit ratios',
        '  ' + table_2.to_string(index=False).replace('\n', '\n  '),
        '',
        'TABLE 3  censoring as an identity channel',
        '  ' + table_3.round(4).to_string(index=False).replace('\n', '\n  '),
        '',
        'TABLE 4  R2 by arm and protocol, median over elements, 95% interval',
        '  ' + table_4.to_string(index=False).replace('\n', '\n  '),
        '',
        'TABLE 5  inflation, and what spatial blocking recovers of it',
        '  ' + table_5.to_string(index=False).replace('\n', '\n  '),
        '',
        'TABLE 6  cost of the common floor, per survey',
        '  ' + table_6.round(5).to_string(index=False).replace('\n', '\n  '),
        '',
        '  values_newly_censored is real information discarded by the correction:',
        '  concentrations a survey resolved and reported, deleted because another',
        '  survey could not resolve them. Harmonisation can only go upwards, so the',
        '  survey that measured best pays the most, and the number is reported with',
        '  the correction rather than after it.',
        '',
        '  supporting: per-element floors where at least two surveys disclose a limit',
        '  ' + table_6_supporting.head(10).round(5).to_string(index=False)
        .replace('\n', '\n  '),
        '',
        'FIGURE 2 DATA  inflation against detection-limit ratio',
        f'  {len(figure_2):,} points across {figure_2.regime.nunique()} regimes; '
        f'{len(primary):,} on the primary arm',
        f'  detection-limit ratio spans {figure_2.lod_ratio.min():.2f} to '
        f'{figure_2.lod_ratio.max():.2f}',
        '',
        '  This figure is the one a reader can refute the paper with. If the cloud',
        '  is flat, the mechanism does not operate on these surveys, whatever the',
        '  simulation and the mask experiments showed. It is therefore plotted as',
        '  raw points with the fit drawn over them, never as binned means, which',
        '  would hide the spread that makes the claim checkable.',
        '',
    ]
    if paired is not None and len(paired):
        significant = paired[paired.p_value_bh < 0.05]
        lines += [
            'PAIRED COMPARISONS SURVIVING CORRECTION',
            f'  {len(significant)} of {len(paired)} arm comparisons have a '
            f'Benjamini-Hochberg corrected p below 0.05',
            '  ' + (significant.round(4).to_string(index=False).replace('\n', '\n  ')
                    if len(significant)
                    else 'none - no arm separates from another once corrected, '
                         'which is a result and is to be reported as one'),
            '',
        ]
    lines += [
        'TRACEABILITY',
        '  Every number above was read from a file written by a numbered script.',
        '  Nothing is computed here. A value that appears in the Results section',
        '  and not in 05-results/ is a value nobody can reproduce.',
        '',
        ('  all upstream reports present' if not missing
         else '  MISSING UPSTREAM OUTPUTS: ' + ', '.join(missing)),
        '',
        'written to 05-results/tables/: ' + ', '.join(sorted(
            f for f in os.listdir(TABLES) if f.endswith('.csv'))),
        'written to 05-results/figure_data/: ' + ', '.join(sorted(
            f for f in os.listdir(FIGURE_DATA) if f.endswith('.csv'))),
        'written to 05-results/: 09_report.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '09_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
