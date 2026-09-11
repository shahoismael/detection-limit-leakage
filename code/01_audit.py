"""
01_audit - the censoring audit and the discordant-pair test.  [E0a, E1c]

Reads the aligned tables written by 00_ingest and produces the two results the
project gate depends on. Nothing downstream of this script runs until both pass.

E0a  Per-element detection limit, censoring rate and flag convention for every
     survey, and the cross-survey detection-limit ratio per shared element. This
     generalises the single-basin audit table of Engle and Brunner (2019) to
     three continental surveys.

E1c  Discordant-censoring pairs. For a pair of measurement channels, a sample is
     discordant when one channel censors it and the other quantifies it. NGSA
     measures the same physical samples under up to three digestions with
     different limits, so discordance is established WITHIN one survey, one
     laboratory and one sampling campaign. Concentration is identical by
     construction, so geology cannot explain the pair. This is the test that
     removes the study's largest threat by design rather than by statistical
     control, and it is why the cross-survey comparison is not load-bearing.

Outputs to 05-results/
    01_lod_table.csv              every element, every survey: LOD, rate, convention
    01_lod_ratio.csv              shared elements ranked by cross-survey LOD ratio
    01_discordant_pairs.csv       NGSA within-survey, per element and method pair
    01_discordant_examples.csv    up to 20 named samples per pair, for inspection
    01_audit_report.txt           the gate verdict

Usage
    python 01_audit.py
    python 01_audit.py --min-ratio 2.0 --min-elements 10
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INTERIM = os.path.join(HERE, '..', '02-datasets', 'interim')
RESULTS = os.path.join(HERE, '..', '05-results')

SURVEYS = ['nasgl', 'ngsa', 'gemas']
FLAG_CONVENTION = {
    'nasgl': 'reported as "<value"; limit read from the flag',
    'ngsa': 'reported as "<value"; limit also declared in the column header',
    'gemas': 'no flag; substituted at LOD/2 before delivery, mask reconstructed',
}


def read_table(base):
    """00_ingest writes parquet when pyarrow is present and gzipped CSV otherwise."""
    for extension, reader in (('.parquet', pd.read_parquet),
                              ('.csv.gz', pd.read_csv)):
        path = os.path.join(INTERIM, base + extension)
        if os.path.exists(path):
            return reader(path)
    sys.exit(f'missing input: {base}.parquet or {base}.csv.gz in {INTERIM}\n'
             f'run 00_ingest.py first')


# ------------------------------------------------------------------ E0a table
def build_lod_table():
    rows = []
    for survey in SURVEYS:
        conc = read_table(f'{survey}_conc')
        mask = read_table(f'{survey}_mask')
        lod = read_table(f'{survey}_lod')
        for element in conc.columns:
            m = mask[element].astype(bool)
            limits = lod[element].dropna().unique()
            measured = conc[element].dropna()
            rows.append({
                'survey': survey.upper(),
                'element': element,
                'n_rows': len(conc),
                'n_measured': int(measured.notna().sum()),
                'n_censored': int(m.sum()),
                'n_missing': int(conc[element].isna().sum() - m.sum()),
                'censored_pct': round(100 * m.mean(), 3),
                'lod': round(float(limits[0]), 8) if len(limits) == 1 else np.nan,
                'lod_distinct_values': len(limits),
                'min_measured': round(float(measured.min()), 8) if len(measured) else np.nan,
                'median_measured': round(float(measured.median()), 8) if len(measured) else np.nan,
                'flag_convention': FLAG_CONVENTION[survey],
            })
    return pd.DataFrame(rows)


def build_lod_ratio(lod_table, min_ratio):
    wide = lod_table.pivot(index='element', columns='survey',
                           values=['lod', 'censored_pct'])
    shared = [e for e in wide.index
              if wide.loc[e, 'lod'].notna().all()
              or wide.loc[e, 'censored_pct'].notna().all()]
    rows = []
    for element in shared:
        limits = {s: wide.loc[element, ('lod', s)] for s in ('NASGL', 'NGSA', 'GEMAS')}
        rates = {s: wide.loc[element, ('censored_pct', s)] for s in ('NASGL', 'NGSA', 'GEMAS')}
        usable = {s: v for s, v in limits.items() if pd.notna(v) and v > 0}
        if len(usable) < 2:
            continue
        lo_survey = min(usable, key=usable.get)
        hi_survey = max(usable, key=usable.get)
        ratio = usable[hi_survey] / usable[lo_survey]
        rows.append({
            'element': element,
            'lod_NASGL': limits['NASGL'], 'lod_NGSA': limits['NGSA'],
            'lod_GEMAS': limits['GEMAS'],
            'censored_pct_NASGL': rates['NASGL'], 'censored_pct_NGSA': rates['NGSA'],
            'censored_pct_GEMAS': rates['GEMAS'],
            'lowest_lod_survey': lo_survey, 'highest_lod_survey': hi_survey,
            'lod_ratio': round(float(ratio), 3),
            'surveys_compared': len(usable),
            'divergent': bool(ratio >= min_ratio),
        })
    return pd.DataFrame(rows).sort_values('lod_ratio', ascending=False)


# ----------------------------------------------------------- E1c discordance
def agreement(va, vb, both, window=None, min_n=30):
    """How closely two measurement channels agree where both quantify.

    Returned as the Spearman correlation of the paired values, the median of
    log10(b / a), and the sample count behind them.

    A global summary is not sufficient and an earlier version of this function
    relied on one. Two channels can agree well across four orders of magnitude
    and still diverge in the lowest decade, which is the only region where
    censoring occurs. Ca under aqua regia against XRF has a global median log10
    ratio of 0.079, comfortably inside any tolerance, yet aqua regia censors
    samples whose XRF value is three times aqua regia's own limit, because aqua
    regia does not dissolve silicate-bound calcium. Agreement is therefore also
    computed inside a window bracketing the detection limits, and the windowed
    figure is what the filter uses.
    """
    x, y = va[both], vb[both]
    keep = (x > 0) & (y > 0)
    if window is not None:
        lo, hi = window
        keep &= (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
    x, y = x[keep], y[keep]
    if len(x) < min_n:
        return np.nan, np.nan, int(len(x))
    rho = float(pd.Series(x).corr(pd.Series(y), method='spearman'))
    bias = float(np.median(np.log10(y.to_numpy() / x.to_numpy())))
    return rho, bias, int(len(x))


def discordant_pairs(max_examples, min_rho, max_bias):
    """NGSA only: the same physical samples measured under several digestions.

    For methods A and B and element e, a row is discordant when e is censored
    under one method and quantified under the other.

    Discordance alone is NOT evidence that censoring carries laboratory identity.
    NGSA's three channels are a near-total digestion, an aqua regia partial
    extraction and a weak mobile-metal-ion leach, and they do not recover the
    same quantity: W under aqua regia and under ICP-MS shares one detection
    limit yet disagrees on 87% of samples, which is chemistry, not a threshold.

    Two filters separate the two explanations.

      COMPARABLE   the pair agrees where both channels quantify - Spearman rho
                   at or above min_rho and a median log10 ratio within max_bias
                   of zero. Only comparable pairs support the identity claim.
      ABOVE-LIMIT  for each discordant sample, whether the value the other
                   channel reported exceeds the censoring channel's own limit.
                   Below it, censoring is the correct behaviour of a coarser
                   instrument on the same concentration - which is precisely the
                   threshold effect this study is about. Above it, the censoring
                   channel failed to report a concentration it should have
                   resolved, which is an instrument or reporting artefact and is
                   counted separately rather than folded into the evidence.
    """
    conc = read_table('ngsa_allmethods_conc')
    mask = read_table('ngsa_allmethods_mask')
    lod = read_table('ngsa_allmethods_lod')
    meta = read_table('ngsa_meta')
    catalogue = read_table('ngsa_method_catalogue')

    by_element = {}
    for row in catalogue.itertuples():
        by_element.setdefault(row.element, []).append(row.method)

    summary, examples = [], []
    for element, methods in sorted(by_element.items()):
        for a, b in itertools.combinations(sorted(set(methods)), 2):
            ka, kb = f'{element}_{a}', f'{element}_{b}'
            if ka not in conc.columns or kb not in conc.columns:
                continue
            ma, mb = mask[ka].astype(bool), mask[kb].astype(bool)
            va, vb = conc[ka], conc[kb]

            a_cens_b_meas = ma & ~mb & vb.notna()
            b_cens_a_meas = mb & ~ma & va.notna()
            both_measured = ~ma & ~mb & va.notna() & vb.notna()
            n_disc = int(a_cens_b_meas.sum() + b_cens_a_meas.sum())

            lod_a = lod[ka].dropna().unique()
            lod_b = lod[kb].dropna().unique()
            lod_a = float(lod_a[0]) if len(lod_a) else np.nan
            lod_b = float(lod_b[0]) if len(lod_b) else np.nan
            ratio = (max(lod_a, lod_b) / min(lod_a, lod_b)
                     if pd.notna(lod_a) and pd.notna(lod_b)
                     and min(lod_a, lod_b) > 0 else np.nan)

            rho_global, bias_global, n_global = agreement(va, vb, both_measured)

            # Agreement inside the decade bracketing the coarser limit, which is
            # the only region where censoring decisions are made.
            coarse = np.nanmax([lod_a, lod_b]) if pd.notna(lod_a) or pd.notna(lod_b) else np.nan
            window = (coarse / 10.0, coarse * 10.0) if pd.notna(coarse) and coarse > 0 else None
            rho_local, bias_local, n_local = agreement(
                va, vb, both_measured, window=window, min_n=15)

            comparable = (pd.notna(rho_local) and rho_local >= min_rho
                          and abs(bias_local) <= max_bias)

            # Split every discordant sample by whether the reported value sits
            # below the censoring channel's own limit, and record how far from
            # that limit it sits. A threshold effect puts the value BELOW the
            # limit: the coarser instrument could not resolve a concentration
            # the finer one could. A value well above the limit means the
            # channel censored something it should have reported, which is
            # differential recovery between digestions, not a threshold.
            below, ratios = 0, []
            for censoring_lod, selector, quantified in (
                    (lod_a, a_cens_b_meas, vb), (lod_b, b_cens_a_meas, va)):
                if pd.isna(censoring_lod) or censoring_lod <= 0:
                    continue
                values = quantified[selector].dropna()
                below += int((values < censoring_lod).sum())
                ratios.extend((values / censoring_lod).tolist())
            above = n_disc - below
            median_ratio = float(np.median(ratios)) if ratios else np.nan

            # Threshold-driven discordance is one-sided: the coarser channel
            # censors and the finer one reports. Discordance running both ways
            # between channels of comparable limits is a per-sample recovery
            # difference, not a threshold.
            n_ab = int(a_cens_b_meas.sum())
            n_ba = int(b_cens_a_meas.sum())
            directionality = (max(n_ab, n_ba) / n_disc) if n_disc else np.nan

            summary.append({
                'element': element, 'method_a': a, 'method_b': b,
                'lod_a': lod_a, 'lod_b': lod_b,
                'lod_ratio': round(ratio, 3) if pd.notna(ratio) else np.nan,
                'n_rows': len(conc),
                'n_censored_a': int(ma.sum()), 'n_censored_b': int(mb.sum()),
                'n_a_censored_b_measured': int(a_cens_b_meas.sum()),
                'n_b_censored_a_measured': int(b_cens_a_meas.sum()),
                'n_discordant': n_disc,
                'discordant_pct': round(100 * n_disc / len(conc), 3),
                'n_both_measured': int(both_measured.sum()),
                'n_agreement_global': n_global,
                'spearman_rho_global': round(rho_global, 3) if pd.notna(rho_global) else np.nan,
                'median_log10_ratio_global': round(bias_global, 3) if pd.notna(bias_global) else np.nan,
                'n_agreement_near_limit': n_local,
                'spearman_rho': round(rho_local, 3) if pd.notna(rho_local) else np.nan,
                'median_log10_ratio': round(bias_local, 3) if pd.notna(bias_local) else np.nan,
                'comparable': bool(comparable),
                'n_below_censoring_lod': below,
                'n_above_censoring_lod': above,
                'threshold_consistent_pct':
                    round(100 * below / n_disc, 2) if n_disc else np.nan,
                'median_value_over_lod':
                    round(median_ratio, 3) if pd.notna(median_ratio) else np.nan,
                'directionality': round(directionality, 3) if pd.notna(directionality) else np.nan,
            })

            for censored_method, quantified_method, selector, quantified, censoring_lod in (
                    (a, b, a_cens_b_meas, vb, lod_a),
                    (b, a, b_cens_a_meas, va, lod_b)):
                index = np.flatnonzero(selector.to_numpy())[:max_examples]
                for i in index:
                    value = quantified.iloc[i]
                    examples.append({
                        'element': element,
                        'censored_under': censored_method,
                        'quantified_under': quantified_method,
                        'sample_id': meta['sample_id'].iloc[i],
                        'site_id': meta['site_id'].iloc[i],
                        'quantified_value': value,
                        'lod_of_censoring_method': censoring_lod,
                        'below_censoring_lod': bool(pd.notna(censoring_lod)
                                                    and value < censoring_lod),
                        'pair_comparable': bool(comparable),
                    })
    return pd.DataFrame(summary), pd.DataFrame(examples)


def main():
    ap = argparse.ArgumentParser(description='Censoring audit and discordant-pair test')
    ap.add_argument('--min-ratio', type=float, default=2.0,
                    help='detection-limit ratio that counts as divergent')
    ap.add_argument('--min-elements', type=int, default=10,
                    help='divergent elements required to pass the E0a gate')
    ap.add_argument('--min-threshold-consistency', type=float, default=80.0,
                    help='percentage of a pair\'s discordant samples that must '
                         'fall below the censoring channel\'s own limit before '
                         'the pair counts as threshold-driven evidence')
    ap.add_argument('--min-discordant', type=int, default=20,
                    help='discordant samples a pair needs to be counted at all')
    ap.add_argument('--max-examples', type=int, default=20,
                    help='named discordant samples written per method pair')
    ap.add_argument('--min-rho', type=float, default=0.80,
                    help='Spearman rho required for two channels to count as '
                         'measuring the same quantity')
    ap.add_argument('--max-bias', type=float, default=0.30,
                    help='largest allowed median log10 ratio between two '
                         'channels (0.30 is a factor of two)')
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    print('building the LOD table ...', end=' ', flush=True)
    lod_table = build_lod_table()
    lod_table.to_csv(os.path.join(RESULTS, '01_lod_table.csv'), index=False)
    print(f'{len(lod_table)} element-survey rows')

    ratio = build_lod_ratio(lod_table, args.min_ratio)
    ratio.to_csv(os.path.join(RESULTS, '01_lod_ratio.csv'), index=False)
    print(f'cross-survey ratios      ... {len(ratio)} shared elements')

    print('discordant pairs (NGSA)  ...', end=' ', flush=True)
    pairs, examples = discordant_pairs(args.max_examples, args.min_rho, args.max_bias)
    pairs.to_csv(os.path.join(RESULTS, '01_discordant_pairs.csv'), index=False)
    examples.to_csv(os.path.join(RESULTS, '01_discordant_examples.csv'), index=False)
    print(f'{len(pairs)} element-method pairs')

    divergent = ratio[ratio.divergent]
    e0a_pass = len(divergent) >= args.min_elements

    # A pair is evidence only if it clears BOTH filters. Counting elements with
    # any discordance is the wrong rule and passes pairs that are measuring a
    # different quantity or censoring values they could resolve.
    comparable = pairs[pairs.comparable]
    rejected = pairs[~pairs.comparable]
    productive = (comparable[comparable.n_discordant > 0]
                  .sort_values('n_discordant', ascending=False))
    qualifying = (pairs[pairs.comparable
                        & (pairs.n_discordant >= args.min_discordant)
                        & (pairs.threshold_consistent_pct >= args.min_threshold_consistency)]
                  .sort_values('n_discordant', ascending=False))
    e1c_pass = qualifying.element.nunique() >= args.min_elements

    # The pairs a global agreement statistic would have wrongly admitted: they
    # agree across their full range but not in the decade where censoring
    # happens, and their discordant samples sit above the censoring channel's
    # own limit. The censoring channel did not fail to resolve a small
    # concentration; it recovered less of that element from that particular
    # sample. This table is the evidence for why the test fails, so it is
    # selected on the global criterion and reported even though every row of it
    # is excluded from the identity evidence.
    globally_comparable = (pairs.spearman_rho_global.notna()
                           & (pairs.spearman_rho_global >= args.min_rho)
                           & (pairs.median_log10_ratio_global.abs() <= args.max_bias))
    recovery_driven = (pairs[globally_comparable
                             & ~pairs.comparable
                             & (pairs.n_discordant >= args.min_discordant)
                             & (pairs.median_value_over_lod > 1.0)]
                       .sort_values('median_value_over_lod', ascending=False))

    top_ratio = ratio.head(12)[['element', 'lod_NASGL', 'lod_NGSA', 'lod_GEMAS',
                                'lod_ratio', 'lowest_lod_survey', 'highest_lod_survey']]
    top_pairs = productive.head(12)[['element', 'method_a', 'method_b', 'lod_ratio',
                                     'spearman_rho', 'n_discordant',
                                     'median_value_over_lod', 'directionality',
                                     'threshold_consistent_pct']]

    lines = [
        '01_audit report',
        '',
        'E0a  CENSORING AUDIT',
        f'  element-survey rows        {len(lod_table)}',
        f'  shared elements compared   {len(ratio)}',
        f'  divergent (ratio >= {args.min_ratio:g}x)     {len(divergent)}',
        '',
        '  widest cross-survey detection-limit ratios',
        '  ' + top_ratio.to_string(index=False).replace('\n', '\n  '),
        '',
        f'  VERDICT E0a: {"PASS" if e0a_pass else "FAIL"} '
        f'({len(divergent)} divergent, {args.min_elements} required)',
        '',
        'E1c  DISCORDANT-CENSORING PAIRS  (NGSA, within survey)',
        '',
        '  Two filters, applied together. A pair is evidence only if the two',
        '  channels agree IN THE DECADE AROUND THE DETECTION LIMIT (so the',
        '  difference between them there is a threshold, not chemistry) AND the',
        '  discordant values sit below the censoring channel\'s own limit (so',
        '  censoring is the coarser instrument failing to resolve a concentration',
        '  the finer one could). Either filter alone admits pairs that prove',
        '  nothing, and a global agreement statistic hides the failure entirely.',
        '',
        '  median_value_over_lod is the diagnostic to read first. Below 1, the',
        '  censored value really was under the limit that censored it. Above 1,',
        '  the channel censored a concentration it should have reported, which is',
        '  differential recovery between digestions.',
        '',
        f'  element-method pairs tested          {len(pairs)}',
        f'  rejected as not comparable           {len(rejected)}   '
        f'(rho < {args.min_rho:g} or |median log10 ratio| > {args.max_bias:g})',
        f'  comparable pairs                     {len(comparable)}',
        f'  comparable pairs with discordance    {len(productive)} '
        f'across {productive.element.nunique()} elements',
        f'  discordant samples, comparable pairs {int(productive.n_discordant.sum()):,}',
        f'  of those, below the censoring limit  '
        f'{int(productive.n_below_censoring_lod.sum()):,} '
        f'({100 * productive.n_below_censoring_lod.sum() / max(productive.n_discordant.sum(), 1):.1f}%)',
        f'  QUALIFYING pairs (both filters)      {len(qualifying)} '
        f'across {qualifying.element.nunique()} elements '
        f'[>= {args.min_discordant} samples, >= {args.min_threshold_consistency:g}% below limit]',
        '',
        '  comparable pairs, ranked by discordance',
        '  ' + (top_pairs.to_string(index=False).replace('\n', '\n  ')
              if len(top_pairs) else 'none'),
        '',
        f'  VERDICT E1c: {"PASS" if e1c_pass else "FAIL"} '
        f'({qualifying.element.nunique()} elements clear both filters, '
        f'{args.min_elements} required)',
        '',
        '  A FAIL here is a result, not a defect to be engineered away. It says',
        '  that NGSA\'s three digestions do not hold concentration constant, so a',
        '  shared physical sample does not by itself isolate a threshold effect.',
        '  Loosening either filter until the test passes would reproduce, inside',
        '  this study, the exact practice the study is about.',
        '',
        '  A FAIL does not weaken E0b. The simulation establishes the causal claim',
        '  over a known generative process and does not depend on NGSA. What a',
        '  FAIL does mean is that the within-survey discordant-pair test cannot',
        '  serve as the identification strategy, and any claim that it removes the',
        '  geology confound by design must be withdrawn.',
        '',
        'RECOVERY-DRIVEN DISCORDANCE  (why the test fails; excluded from evidence)',
        f'  pairs that agree globally but not near the limit, with discordant '
        f'values above it: {len(recovery_driven)}',
        '  ' + (recovery_driven[['element', 'method_a', 'method_b', 'lod_a', 'lod_b',
                                 'spearman_rho_global', 'median_log10_ratio_global',
                                 'spearman_rho', 'n_discordant',
                                 'median_value_over_lod', 'directionality']]
              .to_string(index=False).replace('\n', '\n  ')
              if len(recovery_driven) else 'none'),
        '',
        '  Every row here would have passed a global agreement test. That is the',
        '  point of the table: the failure is invisible until agreement is measured',
        '  in the decade where censoring decisions are actually made.',
        '',
        '  These channels censored samples whose concentration, as reported by the',
        '  other channel, is a multiple of the censoring channel\'s own limit. The',
        '  censoring was not a resolution failure. Aqua regia leaves silicate-bound',
        '  calcium and magnesium undissolved, so a sample can be genuinely below',
        '  the aqua regia limit and far above it by X-ray fluorescence. Where the',
        '  discordance also runs in both directions, the recovered fraction varies',
        '  sample by sample and no fixed offset describes it.',
        '',
        '  This is a property of the digestions, not of the detection limits, and',
        '  it is reported so that it is not mistaken for either.',
        '',
        'written to 05-results/: 01_lod_table.csv, 01_lod_ratio.csv,',
        '  01_discordant_pairs.csv, 01_discordant_examples.csv, 01_audit_report.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '01_audit_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
