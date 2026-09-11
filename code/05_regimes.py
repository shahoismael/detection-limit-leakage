"""
05_regimes - the four censoring-handling regimes, built on one common base  [E2]

Each regime is a different answer to the same question: what do you put where a
survey reported a non-detect? The paper's claim is that the answer changes the
reported cross-region performance, so the regimes have to differ in that choice
and in nothing else.

  (a) substitution      each non-detect becomes half of the limit that censored
                        it, using that survey's own published limit row by row
                        and never a constant fitted across surveys.
  (b) complete_case     every sample carrying any non-detect is deleted, after
                        Shelton et al. (2021), who chose deletion precisely to
                        avoid the imputation bias that (a) introduces.
  (c) mask_feature      the values of (a), with the binary mask added as explicit
                        features, which is the workflow Engle and Brunner (2019)
                        recommend.
  (d) common_floor      one censoring threshold per element, the coarsest of the
                        three surveys, applied to all of them. Values below it
                        are marked censored and left empty for the censored
                        objective in 06_models.py - they are NOT substituted,
                        which is the entire point of the regime.

THE COMMON BASE, AND WHY IT EXISTS

A cell can be empty for two unrelated reasons: the value was censored, or the
element was never measured on that sample. Only the first is this paper's
subject. If a regime had to invent a rule for never-measured cells as well, the
comparison between regimes would confound censoring treatment with missing-data
treatment, and the difference between (a) and (b) would no longer be
attributable to censoring.

So one base is fixed first: rows with no never-measured cell anywhere in the
shared element set. Every regime is built on that identical base. This is itself
a completeness filter and therefore has exactly the survey-dependent retention
E1d is about, which is why the retention is reported per survey rather than
mentioned. It is applied identically to all four regimes, so it cannot generate a
difference between them.

THE FIFTH CHANNEL

Before any regime, variable selection is checked. Engle and Brunner (2019) screen
variables on signal-to-noise with the noise term s = U + DL/3, dropping a
variable when S/N < 0.5. That threshold depends on the detection limit, so the
surviving variable set can differ by survey whenever detection limits differ -
leakage that enters before any handling regime touches the data. The rule is run
independently per survey and the sets compared.

No per-sample analytical uncertainty is distributed with these surveys, so U is
set to zero. That makes the noise term smaller, the ratio larger, and the screen
less likely to drop anything. If the surviving sets still differ by survey under
a screen biased against finding a difference, the finding holds a fortiori.

Outputs to 02-datasets/processed/
    regime_base_meta            base rows, with per-regime membership flags
    regime_substitution_conc    (a), also the value matrix for (c)
    regime_common_floor_conc    (d), empty below the common floor
    regime_common_floor_mask    (d), 1 where the common floor censors
    regime_common_floor_bounds  (d), per-cell AFT upper bound
    regime_base_mask            the original mask on base rows, features for (c)

Outputs to 05-results/
    05_common_floor.csv         per element: each survey's limit, the floor, the
                                extra censoring the floor imposes on each survey
    05_signal_to_noise.csv      the Engle and Brunner screen, per survey
    05_regimes_report.txt

Usage
    python 05_regimes.py
    python 05_regimes.py --sn-threshold 0.5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, '..', '02-datasets', 'processed')
RESULTS = os.path.join(HERE, '..', '05-results')
SEED = 20260906


def read_table(base, folder=PROCESSED):
    for extension, reader in (('.parquet', pd.read_parquet), ('.csv.gz', pd.read_csv)):
        path = os.path.join(folder, base + extension)
        if os.path.exists(path):
            return reader(path)
    sys.exit(f'missing input: {base} in {folder}. Run 04_transform.py first.')


def write_table(frame, base, folder=PROCESSED):
    os.makedirs(folder, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        path = os.path.join(folder, base + '.parquet')
        frame.to_parquet(path, index=False)
    except ImportError:
        path = os.path.join(folder, base + '.csv.gz')
        frame.to_csv(path, index=False, compression='gzip')
    return os.path.basename(path)


def retention_by_survey(surveys, keep, label):
    """Rows kept per survey, which is the quantity E1d showed to be a survey tag."""
    frame = pd.DataFrame({'survey': surveys.to_numpy(), 'keep': np.asarray(keep, int)})
    summary = (frame.groupby('survey').keep
               .agg(n='size', retained='sum')
               .assign(retention_pct=lambda d: (100 * d.retained / d.n).round(2))
               .reset_index())
    summary.insert(0, 'filter', label)
    return summary


def build_common_base(conc, mask, meta):
    """Rows with no never-measured cell. Censored cells are kept - they are the
    subject - but a cell that was never analysed carries no information about
    censoring and would force every regime to invent a missing-data rule."""
    never_measured = conc.isna() & (mask == 0)
    keep = (never_measured.sum(axis=1) == 0)
    return keep, never_measured, retention_by_survey(meta.survey, keep,
                                                     'no never-measured cell')


def substitution_regime(conc, mask, lod):
    """(a) Half the limit that censored the cell, row by row.

    The limit used is the one in force for that survey and element, never a
    pooled constant: a constant fitted across surveys would leak the pooled
    distribution into every row and would also erase the very between-survey
    difference under study.
    """
    filled = conc.copy()
    censored = mask == 1
    half = lod / 2.0
    unresolved = int((censored & half.isna()).sum().sum())
    filled = filled.where(~censored, half)
    return filled, unresolved


def common_floor_regime(conc, mask, lod, meta, elements):
    """(d) One threshold per element, the coarsest of the three surveys.

    Harmonising downwards is impossible - a survey cannot report what its
    instrument could not resolve - so harmonisation is upwards, to the coarsest
    limit. Everything below that floor is censored in every survey, which
    destroys the between-survey difference in WHERE the threshold sits. That is
    the correction, and its cost is real information thrown away in the surveys
    that measured more finely. The cost is quantified here rather than described,
    as the extra censoring the floor imposes on each survey.

    Values below the floor are emptied, not substituted. They are handed to the
    censored objective in 06_models.py with an upper bound, which is what makes
    this regime different from (a) rather than a coarser version of it.
    """
    limits = []
    for element in elements:
        row = {'element': element}
        per_survey = {}
        for survey, index in meta.groupby('survey', sort=False).groups.items():
            values = lod.loc[list(index), element].dropna().unique()
            per_survey[survey] = float(values.max()) if len(values) else np.nan
            row[f'lod_{survey}'] = per_survey[survey]
        known = [v for v in per_survey.values() if np.isfinite(v)]
        row['common_floor'] = float(max(known)) if known else np.nan
        row['n_surveys_with_limit'] = len(known)
        limits.append(row)
    floor_table = pd.DataFrame(limits)
    floor = floor_table.set_index('element').common_floor.reindex(elements)

    values = conc.to_numpy(dtype=float)
    threshold = floor.to_numpy(dtype=float)[None, :]
    with np.errstate(invalid='ignore'):
        below = (values < threshold) & np.isfinite(values) & np.isfinite(threshold)
    already = mask.to_numpy() == 1

    floored_mask = pd.DataFrame((below | already).astype('int8'),
                                columns=elements, index=conc.index)
    floored = conc.mask(floored_mask == 1)
    bounds = pd.DataFrame(np.where(floored_mask.to_numpy() == 1, threshold, np.nan),
                          columns=elements, index=conc.index)

    added = pd.DataFrame(below & ~already, columns=elements, index=conc.index)
    cost = []
    for survey, index in meta.groupby('survey', sort=False).groups.items():
        rows = np.asarray(list(index))
        cost.append({'survey': survey,
                     'censored_pct_before': round(100 * already[rows].mean(), 3),
                     'censored_pct_after': round(
                         100 * floored_mask.loc[rows].to_numpy().mean(), 3),
                     'values_newly_censored': int(added.loc[rows].to_numpy().sum())})
    return floored, floored_mask, bounds, floor_table, pd.DataFrame(cost)


def signal_to_noise_screen(conc, mask, lod, meta, elements, threshold):
    """The Engle and Brunner (2019) variable screen, run per survey.

    s = U + DL/3, drop when S/N < 0.5. U is set to zero because no per-sample
    analytical uncertainty ships with these surveys; that inflates every ratio,
    so the screen drops fewer variables than theirs would and a divergence
    between surveys is a conservative result rather than an artefact of the
    substitution.
    """
    rows = []
    for survey, index in meta.groupby('survey', sort=False).groups.items():
        block = conc.loc[list(index)]
        block_mask = mask.loc[list(index)]
        block_lod = lod.loc[list(index)]
        for element in elements:
            measured = block[element].where(block_mask[element] == 0).dropna()
            limits = block_lod[element].dropna().unique()
            detection_limit = float(limits.max()) if len(limits) else np.nan
            signal = float(measured.median()) if len(measured) else np.nan
            noise = detection_limit / 3.0 if np.isfinite(detection_limit) else np.nan
            ratio = signal / noise if (np.isfinite(noise) and noise > 0) else np.inf
            # The uncertainty at which this element would first be dropped.
            # S/(U + DL/3) = threshold solves to U = S/threshold - DL/3, so the
            # screen that retains everything at U = 0 becomes a statement about
            # how much analytical uncertainty it would take to bite. This is the
            # only way to say anything about a rule whose input the surveys do
            # not publish, and it invents no value.
            if np.isfinite(signal) and np.isfinite(noise):
                breaking_uncertainty = signal / threshold - noise
            else:
                breaking_uncertainty = np.nan
            rows.append({'survey': survey, 'element': element,
                         'median_measured': signal,
                         'detection_limit': detection_limit,
                         'noise_s': noise,
                         'signal_to_noise': ratio,
                         'uncertainty_that_drops_it': breaking_uncertainty,
                         'limit_known': bool(np.isfinite(detection_limit)),
                         'retained': bool(not np.isfinite(ratio) or ratio >= threshold)})
    table = pd.DataFrame(rows)
    sets = {survey: set(block[block.retained].element)
            for survey, block in table.groupby('survey')}
    # An element a survey never censored has no limit in the delivered file, so
    # its ratio is infinite and the screen keeps it. That is not the same as a
    # confidently high ratio, and it inflates retention for whichever survey
    # discloses the fewest limits - GEMAS above all, whose limits exist here only
    # where 00_ingest.py could reconstruct them. Counted so that a divergence in
    # the surviving sets is never read as fine analysis when it is silence.
    unknown = (table[~table.limit_known].groupby('survey').element
               .agg(n_limits_unknown='size').reset_index())
    return table, sets, unknown


def uncertainty_at_which_the_screen_bites(table):
    """At what analytical uncertainty the screen starts dropping, and diverging.

    At uncertainty U an element survives in a survey while U < that survey's
    breaking value. The screen therefore starts acting at the smallest breaking
    value anywhere, and the surviving SETS start to differ between surveys at the
    smallest breaking value belonging to an element whose surveys do not all
    break together. The second number is the one that matters: a rule that drops
    the same elements everywhere is a rule, while a rule that drops different
    elements in different surveys is a leakage channel.
    """
    rows = []
    for element, block in table.groupby('element'):
        breaks = block.set_index('survey').uncertainty_that_drops_it.dropna()
        if breaks.empty:
            continue
        # A survey with no disclosed limit never breaks, so it always disagrees
        # with one that does.
        silent = len(block) - len(breaks)
        rows.append({'element': element,
                     'first_drop_at_uncertainty': float(breaks.min()),
                     'last_drop_at_uncertainty': float(breaks.max()),
                     'surveys_that_never_drop_it': silent,
                     'diverges': bool(silent > 0 or breaks.min() < breaks.max())})
    summary = pd.DataFrame(rows).sort_values('first_drop_at_uncertainty')
    diverging = summary[summary.diverges]
    return (summary,
            float(summary.first_drop_at_uncertainty.min()) if len(summary) else np.nan,
            float(diverging.first_drop_at_uncertainty.min()) if len(diverging) else np.nan)


def main():
    ap = argparse.ArgumentParser(description='Four censoring-handling regimes')
    ap.add_argument('--sn-threshold', type=float, default=0.5,
                    help='Engle and Brunner drop a variable below this ratio')
    ap.add_argument('--min-rows-per-survey', type=int, default=100,
                    help='rows a regime must keep in EVERY survey to be usable '
                         'under leave-one-survey-out')
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    meta = read_table('pooled_meta')
    conc = read_table('pooled_conc')
    mask = read_table('pooled_mask')
    lod = read_table('pooled_lod')
    elements = list(conc.columns)
    print(f'pooled {len(conc):,} rows x {len(elements)} elements')

    keep, never_measured, base_retention = build_common_base(conc, mask, meta)
    meta_base = meta.loc[keep].reset_index(drop=True)
    conc_base = conc.loc[keep].reset_index(drop=True)
    mask_base = mask.loc[keep].reset_index(drop=True)
    lod_base = lod.loc[keep].reset_index(drop=True)
    print(f'common base: {len(conc_base):,} rows '
          f'({100 * len(conc_base) / len(conc):.1f}%)')

    substituted, unresolved = substitution_regime(conc_base, mask_base, lod_base)
    if unresolved:
        sys.exit(f'{unresolved} censored cells have no detection limit, so LOD/2 is '
                 'undefined for them. Fix 00_ingest.py rather than filling them with '
                 'a pooled guess.')

    complete_case = (mask_base.sum(axis=1) == 0)
    complete_retention = retention_by_survey(meta_base.survey, complete_case,
                                             'complete-case deletion')
    # A regime that empties a survey cannot be compared across surveys at all, and
    # one with too few rows cannot be cross-validated. Both are outcomes of the
    # remedy rather than failures here, so they are flagged and carried forward
    # instead of being repaired.
    complete_case_viable = bool(
        (complete_retention.retained >= args.min_rows_per_survey).all())

    floored, floored_mask, bounds, floor_table, floor_cost = common_floor_regime(
        conc_base, mask_base, lod_base, meta_base, elements)

    sn_table, sn_sets, sn_unknown = signal_to_noise_screen(
        conc_base, mask_base, lod_base, meta_base, elements, args.sn_threshold)
    survey_names = sorted(sn_sets)
    common_survivors = set.intersection(*sn_sets.values()) if sn_sets else set()
    any_survivor = set.union(*sn_sets.values()) if sn_sets else set()
    screen_diverges = bool(any_survivor - common_survivors)
    bite_table, first_drop, first_divergence = uncertainty_at_which_the_screen_bites(
        sn_table)
    bite_table.round(6).to_csv(
        os.path.join(RESULTS, '05_screen_uncertainty_thresholds.csv'), index=False)

    meta_out = meta_base.copy()
    meta_out['in_complete_case'] = complete_case.astype('int8').to_numpy()

    written = [write_table(meta_out, 'regime_base_meta'),
               write_table(substituted, 'regime_substitution_conc'),
               write_table(mask_base, 'regime_base_mask'),
               write_table(floored, 'regime_common_floor_conc'),
               write_table(floored_mask, 'regime_common_floor_mask'),
               write_table(bounds, 'regime_common_floor_bounds')]
    floor_table.round(6).to_csv(os.path.join(RESULTS, '05_common_floor.csv'),
                                index=False)
    # The per-survey cost, written out and not only printed. It is the number the
    # correction is judged by - how much real information each survey loses - and
    # an earlier version left it in the report text only, so 09_report.py had
    # nothing to read and built its cost table from the per-element limits
    # instead, under a caption that promised per-survey costs.
    floor_cost.to_csv(os.path.join(RESULTS, '05_common_floor_cost.csv'), index=False)
    sn_table.round(4).to_csv(os.path.join(RESULTS, '05_signal_to_noise.csv'),
                             index=False)

    divergent_floor = floor_table[floor_table.n_surveys_with_limit >= 2].copy()
    spread = []
    for row in divergent_floor.itertuples():
        limits = [getattr(row, f'lod_{s}') for s in survey_names]
        limits = [v for v in limits if np.isfinite(v)]
        spread.append(max(limits) / min(limits) if min(limits) > 0 else np.nan)
    divergent_floor['limit_ratio'] = spread
    divergent_floor = divergent_floor.sort_values('limit_ratio', ascending=False)

    lines = [
        '05_regimes report',
        '',
        f'pooled input      {len(conc):,} rows x {len(elements)} elements',
        f'common base       {len(conc_base):,} rows '
        f'({100 * len(conc_base) / len(conc):.1f}% of pooled)',
        '',
        'THE COMMON BASE',
        '  ' + base_retention.to_string(index=False).replace('\n', '\n  '),
        '',
        '  Rows are dropped here only for cells that were never measured, never for',
        '  censored cells. Retention differs by survey, which is E1d operating on',
        '  this study rather than on Shelton et al. The filter is identical across',
        '  all four regimes, so it cannot produce a difference between them.',
        '',
        'REGIME SIZES',
        f'  (a) substitution     {len(substituted):,} rows, all base rows retained',
        f'  (b) complete_case    {int(complete_case.sum()):,} rows '
        f'({100 * complete_case.mean():.1f}% of base)',
        '  ' + complete_retention.to_string(index=False).replace('\n', '\n  '),
        f'      viable for modelling: {"yes" if complete_case_viable else "NO"} '
        f'(needs >= {args.min_rows_per_survey} rows in EVERY survey, not in total:',
        '      leave-one-survey-out holds a whole survey out, so a regime that keeps',
        '      one survey and empties the others cannot be validated across regions',
        '      at all, whatever its pooled row count looks like)',
        f'  (c) mask_feature     {len(substituted):,} rows, '
        f'{len(elements)} values + {len(elements)} mask columns',
        f'  (d) common_floor     {len(floored):,} rows, '
        f'{int(floored_mask.to_numpy().sum()):,} censored cells after flooring',
        '',
        '  Sample counts are NOT matched across regimes, and cannot be: deletion',
        '  removes rows by construction. 07_validate.py therefore reports every',
        '  regime both on its own rows and restricted to the intersection, so that',
        '  a difference in performance is never silently a difference in n.',
        '',
        '  Complete-case deletion scales with the number of features, not with the',
        '  censoring rate of any one of them. Shelton et al. (2021) applied it over',
        f'  seven and nine features; the downstream task here uses {len(elements) - 1},',
        '  since every element except the rotating target is a predictor. If the',
        '  regime retains too few samples or empties a survey, that is a result about',
        '  the remedy rather than a defect in this script: deletion does not survive',
        '  continental-scale multi-element data, and any comparison that reports it',
        '  as viable is reporting on a much smaller feature set than it appears to.',
        '',
        'COST OF THE COMMON FLOOR',
        '  ' + floor_cost.to_string(index=False).replace('\n', '\n  '),
        '',
        '  This is the correction being paid for. Harmonisation can only go upwards,',
        '  so every survey that measured more finely than the coarsest one loses the',
        '  values it resolved below that limit. The count above is real information',
        '  discarded, and it is reported with the correction rather than after it.',
        '',
        '  widest limit disparities driving the floor',
        '  ' + divergent_floor.head(10).round(5).to_string(index=False)
        .replace('\n', '\n  '),
        '',
        f'VARIABLE SELECTION AS A FIFTH CHANNEL (Engle and Brunner screen, '
        f'S/N < {args.sn_threshold})',
    ]
    for survey in survey_names:
        lines.append(f'  {survey:8} retains {len(sn_sets[survey])} of '
                     f'{len(elements)} elements')
    for survey in survey_names:
        only_here = sorted(sn_sets[survey] - common_survivors)
        dropped = sorted(set(elements) - sn_sets[survey])
        lines.append(f'  {survey:8} drops: {", ".join(dropped) if dropped else "none"}')
        if only_here:
            lines.append(f'  {survey:8} keeps but at least one other survey drops: '
                         f'{", ".join(only_here)}')
    lines += [
        '',
        '  elements with no disclosed limit, which the screen keeps by default',
        '  ' + sn_unknown.to_string(index=False).replace('\n', '\n  '),
        '  An unknown limit gives an infinite ratio and automatic retention. That is',
        '  silence, not sensitivity, and it favours whichever survey publishes least.',
        '',
        f'  VERDICT E2-selection at U = 0: '
        f'{"LEAKAGE CHANNEL" if screen_diverges else "NOT EVALUABLE"}',
        f'    elements retained by every survey     {len(common_survivors)}',
        f'    elements retained by some but not all {len(any_survivor - common_survivors)}',
        '',
        '  Read a non-divergence here as an underpowered test, not as a negative',
        '  result. With U = 0 the ratio is 3S/DL, which almost nothing fails, and',
        '  an undisclosed limit is retained automatically. The screen cannot be',
        '  evaluated on data that does not carry the uncertainty it consumes.',
        '',
        '  WHAT CAN BE SAID: the uncertainty at which the screen would bite',
        f'    smallest U that drops any element anywhere   {first_drop:.4g} mg/kg',
        f'    smallest U at which the surviving sets differ {first_divergence:.4g} mg/kg',
        '  ' + bite_table.head(8).round(5).to_string(index=False).replace('\n', '\n  '),
        '',
        '  Above that second value the surviving variable set depends on which',
        '  survey a sample came from, which is a leakage channel opening before any',
        '  handling regime is chosen, and no regime downstream can close it. Whether',
        '  real analytical uncertainty reaches it is a question for the laboratories,',
        '  not for this dataset, and it is left stated rather than guessed.',
        '',
        'NOT DONE HERE',
        '  Nothing is transformed. Regimes fill values in composition space; the',
        '  frozen contrast matrix from 04_transform.py is applied in 06_models.py,',
        '  in that order and never the reverse.',
        '',
        'written to 02-datasets/processed/: ' + ', '.join(written),
        'written to 05-results/: 05_common_floor.csv, 05_common_floor_cost.csv,',
        '  05_signal_to_noise.csv, 05_screen_uncertainty_thresholds.csv,',
        '  05_regimes_report.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '05_regimes_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
