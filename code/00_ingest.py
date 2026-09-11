"""
00_ingest - raw survey files to aligned concentration / mask / detection-limit tables.

No values are altered here. Non-detects are separated from concentrations, never
substituted. The three outputs per survey are aligned row-for-row and
column-for-column:

    <survey>_meta.csv.gz   id, survey, latitude, longitude, plus survey-specific keys
    <survey>_conc.csv.gz   measured concentration, NaN where the value is censored
    <survey>_mask.csv.gz   1 where the value is a non-detect, 0 where it is measured
    <survey>_lod.csv.gz    the detection limit that applied to that cell

Reading order matters and is fixed: a censored cell has NaN in _conc and its
threshold in _lod. Nothing downstream may read a concentration without reading the
mask beside it.

Also written, to 05-results/:
    00_shared_elements.csv     the A3 gate - elements common to all three surveys
    00_method_map.csv          which digestion/instrument was taken per element
    00_gemas_lod_recovery.csv  reconstructed GEMAS limits, for manual verification
    00_ingest_report.txt       record counts, censoring rates, gate outcome

Usage
    python 00_ingest.py
    python 00_ingest.py --ngsa-method-order "AR,ICP-MS,XRF,FA,ISE,MMI-ME"
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '02-datasets')
INTERIM = os.path.join(DATA, 'interim')
RESULTS = os.path.join(HERE, '..', '05-results')

NASGL_FILE = os.path.join(DATA, 'NASGL', 'Appendix_3b_Ahorizon_18Sept2013.txt')
NGSA_FILE = os.path.join(DATA, 'NGSA', 'Rec2011_020_110706.csv')
GEMAS_FILE = os.path.join(DATA, 'GEMAS', 'GEMAS_Ap_AquaRegia_XRF.csv')

# ---------------------------------------------------------------- assumptions
# Stated here rather than buried in the code, because each one is a decision a
# reviewer may want to overturn.
#
# A. Stratum. NASGL supplies one A-horizon record per site. NGSA supplies six
#    (three grain sizes x two depths), so one is selected: top outlet sediment at
#    <2 mm, the closest match to the NASGL <2 mm A-horizon and to the GEMAS
#    agricultural topsoil. Overridable with --ngsa-depth / --ngsa-grainsize.
# B. Digestion. The three surveys do not share one. NGSA reports the same element
#    under up to three digestions; one is taken for the shared matrix, in the
#    preference order below, and ALL of them are retained separately for the
#    within-survey discordant-censoring test. GEMAS aqua regia is taken over the
#    XRF block because it covers more of the shared elements.
# C. GEMAS censoring is reconstructed, not read. See recover_gemas_mask.
NGSA_DEFAULT_METHOD_ORDER = ['ICP-MS', 'XRF', 'FA', 'ISE', 'AR', 'MMI-ME']

ELEMENTS = {
    'Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 'Co', 'Cr',
    'Cs', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'Hg', 'In', 'K', 'La', 'Li', 'Mg', 'Mn',
    'Mo', 'Na', 'Nb', 'Ni', 'P', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'S', 'Sb', 'Sc',
    'Se', 'Sn', 'Sr', 'Ta', 'Te', 'Th', 'Ti', 'Tl', 'U', 'V', 'W', 'Y', 'Zn', 'Zr',
}


UNIT_FACTORS_TO_MG_PER_KG = {
    'wt. %': 1e4, 'wt.%': 1e4, 'wt %': 1e4, '%': 1e4, 'percent': 1e4,
    'mg/kg': 1.0, 'ppm': 1.0, 'mg kg-1': 1.0,
    'ug/kg': 1e-3, 'µg/kg': 1e-3, 'ppb': 1e-3,
}


def unit_factor_to_mg_per_kg(unit):
    """Return (multiplier, harmonised unit name) for a reported unit string.

    Every cross-survey quantity in this study is a ratio between detection
    limits, so a unit mismatch does not shift a comparison, it fabricates one.
    An unrecognised unit is left untouched and labelled, never silently assumed
    to be mg/kg.
    """
    key = str(unit).strip().lower()
    for name, factor in UNIT_FACTORS_TO_MG_PER_KG.items():
        if key == name.lower():
            return factor, 'mg/kg'
    if key in ('', 'nan', 'none'):
        return 1.0, 'unstated (assumed mg/kg)'
    return 1.0, f'unconverted ({unit})'


def parse_censored(series):
    """Split a column of strings into (value, is_censored, threshold).

    '<0.03' is a non-detect at 0.03. '0.018' is a measurement. Anything that
    parses as neither becomes missing in all three outputs, and missing is not
    the same thing as censored.
    """
    s = series.astype(str).str.strip()
    censored = s.str.startswith('<')
    numeric_text = s.str.lstrip('<')
    value = pd.to_numeric(numeric_text, errors='coerce')

    conc = value.where(~censored)
    lod = value.where(censored)
    mask = censored & value.notna()
    conc[~mask & value.isna()] = np.nan
    return conc, mask.astype('int8'), lod


def blank_frames(index, columns):
    conc = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    mask = pd.DataFrame(0, index=index, columns=columns, dtype='int8')
    lod = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    return conc, mask, lod


# --------------------------------------------------------------------- NASGL
def load_nasgl():
    """Soil A horizon, conterminous USA. Header sits on file line 13, a units row
    on line 14, data from line 15."""
    raw = pd.read_csv(NASGL_FILE, sep='\t', header=None, skiprows=12,
                      dtype=str, encoding='latin-1', low_memory=False)
    header = raw.iloc[0].str.strip().tolist()
    units = raw.iloc[1].str.strip().tolist()
    body = raw.iloc[2:].reset_index(drop=True)
    body.columns = header

    element_cols = {}
    for col in header:
        if not isinstance(col, str) or not col.startswith('A_'):
            continue
        symbol = col[2:].strip()
        if symbol in ELEMENTS:                       # excludes mineralogy and C_Org etc.
            element_cols[symbol] = col

    meta = pd.DataFrame({
        'survey': 'NASGL',
        'sample_id': body['A_LabID'].str.strip(),
        'site_id': body['SiteID'].str.strip(),
        'latitude': pd.to_numeric(body['Latitude'], errors='coerce'),
        'longitude': pd.to_numeric(body['Longitude'], errors='coerce'),
        'stratum': 'A horizon, <2 mm',
    })

    symbols = sorted(element_cols)
    conc, mask, lod = blank_frames(body.index, symbols)
    unit_by_symbol = dict(zip(header, units))
    provenance = []
    for symbol in symbols:
        col = element_cols[symbol]
        c, m, l = parse_censored(body[col])

        # Unit harmonisation. NASGL reports major elements in weight per cent and
        # trace elements in mg/kg; NGSA and GEMAS report everything in mg/kg.
        # Left unconverted, Fe reads as a detection limit of 0.01 against NGSA's
        # 100 and produces a spurious cross-survey ratio of 10,000 that is
        # nothing but the conversion factor. Every comparison in this study is a
        # ratio of detection limits, so the conversion is not cosmetic.
        unit = str(unit_by_symbol.get(col, '')).strip()
        factor, unit_out = unit_factor_to_mg_per_kg(unit)
        conc[symbol], mask[symbol], lod[symbol] = c * factor, m, l * factor

        limits = lod[symbol].dropna().unique()
        provenance.append({
            'survey': 'NASGL', 'element': symbol, 'source_column': col,
            'method': 'four-acid near-total, ICP-AES/ICP-MS',
            'unit_reported': unit or 'unstated',
            'unit_factor_to_mg_per_kg': factor,
            'unit': unit_out,
            'lod_reported': limits[0] if len(limits) == 1 else np.nan,
            'lod_distinct_values': len(limits),
            'censored_n': int(m.sum()),
            'censored_pct': round(100 * m.mean(), 3),
        })
    return meta, conc, mask, lod, pd.DataFrame(provenance)


# ---------------------------------------------------------------------- NGSA
NGSA_HEADER_RE = re.compile(r'^([A-Z][a-z]?)\s+(\S+)\s+(\S+)\s+(\S+)$')


def parse_ngsa_lod(text):
    """'0.03' -> 0.03. '100-100K' is a lower-upper pair; take the lower."""
    lower = text.split('-')[0].strip()
    multiplier = 1.0
    if lower.upper().endswith('K'):
        lower, multiplier = lower[:-1], 1000.0
    try:
        return float(lower) * multiplier
    except ValueError:
        return np.nan


def load_ngsa(method_order, depth, grain_size):
    """National Geochemical Survey of Australia. Header on file line 12, data
    from line 13, and the same element measured under up to three digestions."""
    raw = pd.read_csv(NGSA_FILE, header=None, skiprows=11, dtype=str,
                      encoding='latin-1', low_memory=False)
    header = raw.iloc[0].str.strip().tolist()
    body = raw.iloc[1:].reset_index(drop=True)
    body.columns = header
    body = body[(body['DEPTH'].str.strip() == depth) &
                (body['GRAIN SIZE'].str.strip() == grain_size)].reset_index(drop=True)

    # every element/method column, kept in full for the discordant-pair test
    catalogue = []
    for col in header:
        match = NGSA_HEADER_RE.match(col.strip()) if isinstance(col, str) else None
        if match and match.group(1) in ELEMENTS:
            symbol, method, unit, lod_text = match.groups()
            catalogue.append({'element': symbol, 'method': method, 'unit': unit,
                              'lod_reported': parse_ngsa_lod(lod_text),
                              'source_column': col})
    catalogue = pd.DataFrame(catalogue)

    meta = pd.DataFrame({
        'survey': 'NGSA',
        'sample_id': body['SAMPLEID'].str.strip(),
        'site_id': body['SITEID'].str.strip(),
        'latitude': pd.to_numeric(body['LATITUDE'], errors='coerce'),
        'longitude': pd.to_numeric(body['LONGITUDE'], errors='coerce'),
        'stratum': f'{depth}, {grain_size}',
    })

    # every method, for E1c
    all_conc, all_mask, all_lod = blank_frames(
        body.index, [f"{r.element}_{r.method}" for r in catalogue.itertuples()])
    for row in catalogue.itertuples():
        key = f'{row.element}_{row.method}'
        c, m, l = parse_censored(body[row.source_column])
        all_conc[key], all_mask[key] = c, m
        all_lod[key] = l.fillna(row.lod_reported)

    # one method per element, for the shared matrix
    rank = {m: i for i, m in enumerate(method_order)}
    catalogue['rank'] = catalogue.method.map(lambda m: rank.get(m, len(rank)))
    chosen = (catalogue.sort_values(['element', 'rank'])
                       .drop_duplicates('element', keep='first'))

    symbols = sorted(chosen.element)
    conc, mask, lod = blank_frames(body.index, symbols)
    provenance = []
    for row in chosen.itertuples():
        key = f'{row.element}_{row.method}'
        conc[row.element] = all_conc[key]
        mask[row.element] = all_mask[key]
        lod[row.element] = all_lod[key]
        provenance.append({
            'survey': 'NGSA', 'element': row.element,
            'source_column': row.source_column, 'method': row.method,
            'unit_reported': row.unit,
            'unit_factor_to_mg_per_kg': unit_factor_to_mg_per_kg(row.unit)[0],
            'unit': unit_factor_to_mg_per_kg(row.unit)[1],
            'lod_reported': row.lod_reported,
            'lod_distinct_values': 1,
            'censored_n': int(all_mask[key].sum()),
            'censored_pct': round(100 * all_mask[key].mean(), 3),
        })
    return (meta, conc[symbols], mask[symbols], lod[symbols],
            pd.DataFrame(provenance), all_conc, all_mask, all_lod, catalogue)


# --------------------------------------------------------------------- GEMAS
GEMAS_MIN_SPIKE_SHARE = 0.01       # a repeated constant below this is not a spike
GEMAS_MIN_DOMINANCE = 10.0         # modal count over the next-most-frequent count


def recover_gemas_mask(values):
    """GEMAS ships with non-detects already replaced by half the detection limit
    and the qualifier removed, so the mask must be reconstructed.

    The detectable signature is modal dominance, not arithmetic. A substitution
    constant is one number written into thousands of cells, while genuine
    measurements repeat almost never: in the Re column 0.00025 appears 1,619
    times and the next most frequent value appears twice, a ratio of 810 to 1.

    An earlier version of this function tested whether the next distinct value
    above the candidate was exactly twice it, and found nothing. That test is
    unsound: the detection limit itself need never appear in the data, because
    no sample has to measure at exactly the limit. In Re the spike is 0.00025
    and the limit is 0.0005, but the next value present is 0.00056.

    A candidate must therefore be modal by a wide margin, cover at least a
    minimum share of the column, and BE THE COLUMN MINIMUM, since nothing can
    sit below a constant that replaced everything under the limit.

    That last test replaces an earlier one requiring the candidate to fall below
    the median, which was wrong in the worst possible direction. Where more than
    half a column is substituted the spike IS the median, so the rule discarded
    precisely the most heavily censored elements: GEMAS Re, 76.6 % substituted
    and the clearest case in the dataset, was rejected by it. The minimum test
    is also strictly stronger - it is the property a substitution constant must
    have and a genuine measurement need not.

    The recovered limit is asserted as twice the spike and written out for
    manual comparison against the published GEMAS limits before any downstream
    use.
    """
    v = pd.to_numeric(values, errors='coerce')
    finite = v.dropna()
    if len(finite) < 20:
        return None

    counts = finite.value_counts()
    if len(counts) < 2:
        return None

    candidate = float(counts.index[0])
    repeats = int(counts.iloc[0])
    runner_up = int(counts.iloc[1])
    share = repeats / len(v)
    dominance = repeats / max(runner_up, 1)

    if share < GEMAS_MIN_SPIKE_SHARE:
        return None
    if dominance < GEMAS_MIN_DOMINANCE:
        return None
    if candidate != float(finite.min()):
        return None
    if candidate <= 0:
        return None

    return {'spike_value': candidate,
            'lod_recovered': 2.0 * candidate,
            'share': share,
            'repeats': repeats,
            'runner_up_repeats': runner_up,
            'dominance': dominance,
            'values_below_spike': int((finite < candidate).sum())}


def load_gemas():
    body = pd.read_csv(GEMAS_FILE, dtype=str, encoding='utf-8-sig', low_memory=False)
    body.columns = [c.strip() for c in body.columns]

    def symbol_of(col):
        base = col[:-1] if col.endswith('_') else col     # AS_ -> AS, IN_ -> IN
        name = base.capitalize()
        return name if name in ELEMENTS else None

    xrf_cols = {c for c in body.columns if c.endswith('_XRF')}
    element_cols = {}
    for col in body.columns:
        if col in xrf_cols or col.startswith(('F1', 'SUSCEPT', 'geom')):
            continue
        symbol = symbol_of(col)
        if symbol and symbol not in element_cols:
            element_cols[symbol] = col

    meta = pd.DataFrame({
        'survey': 'GEMAS',
        'sample_id': body['ID'].str.strip(),
        'site_id': body['C_ID'].str.strip(),
        'latitude': pd.to_numeric(body['YCOO'], errors='coerce'),
        'longitude': pd.to_numeric(body['XCOO'], errors='coerce'),
        'stratum': 'agricultural soil (Ap), 0-20 cm',
    })

    symbols = sorted(element_cols)
    conc, mask, lod = blank_frames(body.index, symbols)
    provenance, recovery = [], []
    for symbol in symbols:
        col = element_cols[symbol]
        values = pd.to_numeric(body[col], errors='coerce')
        found = recover_gemas_mask(values)

        if found is None:
            conc[symbol] = values
            censored_n, censored_pct, lod_value = 0, 0.0, np.nan
        else:
            is_spike = values == found['spike_value']
            conc[symbol] = values.where(~is_spike)
            mask[symbol] = is_spike.astype('int8')
            lod[symbol] = np.where(is_spike, found['lod_recovered'], np.nan)
            censored_n = int(is_spike.sum())
            censored_pct = round(100 * float(is_spike.mean()), 3)
            lod_value = found['lod_recovered']
            recovery.append({'element': symbol, 'source_column': col,
                             'spike_value': found['spike_value'],
                             'lod_recovered': lod_value,
                             'share_at_spike': round(found['share'], 4),
                             'repeats_at_spike': found['repeats'],
                             'runner_up_repeats': found['runner_up_repeats'],
                             'dominance_ratio': round(found['dominance'], 1),
                             'values_below_spike': found['values_below_spike'],
                             'verified_against_published_lod': 'PENDING'})

        provenance.append({
            'survey': 'GEMAS', 'element': symbol, 'source_column': col,
            'method': 'aqua regia, ICP-MS',
            'unit_reported': 'mg/kg (asserted from the GEMAS aqua regia release, '
                             'not carried in the file)',
            'unit_factor_to_mg_per_kg': 1.0, 'unit': 'mg/kg',
            'lod_reported': lod_value,
            'lod_distinct_values': 0 if found is None else 1,
            'censored_n': censored_n, 'censored_pct': censored_pct,
        })
    return meta, conc, mask, lod, pd.DataFrame(provenance), pd.DataFrame(recovery)


# ---------------------------------------------------------------------- write
def write_table(frame, path_base):
    """Parquet when pyarrow is installed, gzipped CSV otherwise. Both are exact;
    the fallback exists so a missing optional dependency cannot stop the run."""
    try:
        import pyarrow  # noqa: F401
        path = path_base + '.parquet'
        frame.to_parquet(path, index=False)
    except Exception:
        path = path_base + '.csv.gz'
        frame.to_csv(path, index=False, compression='gzip')
    return os.path.basename(path)


def main():
    ap = argparse.ArgumentParser(description='Ingest the three continental surveys')
    ap.add_argument('--ngsa-method-order',
                    default=','.join(NGSA_DEFAULT_METHOD_ORDER),
                    help='digestion preference for the shared matrix')
    ap.add_argument('--ngsa-depth', default='TOS')
    ap.add_argument('--ngsa-grainsize', default='<2 mm')
    args = ap.parse_args()

    for folder in (INTERIM, RESULTS):
        os.makedirs(folder, exist_ok=True)

    missing = [p for p in (NASGL_FILE, NGSA_FILE, GEMAS_FILE) if not os.path.exists(p)]
    if missing:
        sys.exit('missing input file(s):\n  ' + '\n  '.join(missing))

    print('reading NASGL  ...', end=' ', flush=True)
    n_meta, n_conc, n_mask, n_lod, n_prov = load_nasgl()
    print(f'{len(n_meta):,} rows, {n_conc.shape[1]} elements')

    print('reading NGSA   ...', end=' ', flush=True)
    (g_meta, g_conc, g_mask, g_lod, g_prov,
     ngsa_all_conc, ngsa_all_mask, ngsa_all_lod, ngsa_catalogue) = load_ngsa(
        [m.strip() for m in args.ngsa_method_order.split(',')],
        args.ngsa_depth, args.ngsa_grainsize)
    print(f'{len(g_meta):,} rows, {g_conc.shape[1]} elements, '
          f'{ngsa_all_conc.shape[1]} element-method columns retained')

    print('reading GEMAS  ...', end=' ', flush=True)
    e_meta, e_conc, e_mask, e_lod, e_prov, e_recovery = load_gemas()
    print(f'{len(e_meta):,} rows, {e_conc.shape[1]} elements, '
          f'{len(e_recovery)} with a reconstructed mask')

    surveys = {
        'nasgl': (n_meta, n_conc, n_mask, n_lod),
        'ngsa': (g_meta, g_conc, g_mask, g_lod),
        'gemas': (e_meta, e_conc, e_mask, e_lod),
    }
    written = []
    for name, (meta, conc, mask, lod) in surveys.items():
        for kind, frame in (('meta', meta), ('conc', conc),
                            ('mask', mask), ('lod', lod)):
            written.append(write_table(
                frame.reset_index(drop=True),
                os.path.join(INTERIM, f'{name}_{kind}')))
    for kind, frame in (('conc', ngsa_all_conc), ('mask', ngsa_all_mask),
                        ('lod', ngsa_all_lod)):
        written.append(write_table(
            frame.reset_index(drop=True),
            os.path.join(INTERIM, f'ngsa_allmethods_{kind}')))
    written.append(write_table(ngsa_catalogue,
                               os.path.join(INTERIM, 'ngsa_method_catalogue')))

    provenance = pd.concat([n_prov, g_prov, e_prov], ignore_index=True)
    provenance.to_csv(os.path.join(RESULTS, '00_method_map.csv'), index=False)
    e_recovery.to_csv(os.path.join(RESULTS, '00_gemas_lod_recovery.csv'), index=False)

    # ------------------------------------------------------------- A3 gate
    sets = {'NASGL': set(n_conc.columns), 'NGSA': set(g_conc.columns),
            'GEMAS': set(e_conc.columns)}
    shared = sorted(set.intersection(*sets.values()))
    rows = []
    for element in shared:
        row = {'element': element}
        for survey, prov in (('NASGL', n_prov), ('NGSA', g_prov), ('GEMAS', e_prov)):
            hit = prov[prov.element == element]
            row[f'{survey}_lod'] = hit.lod_reported.iloc[0] if len(hit) else np.nan
            row[f'{survey}_censored_pct'] = hit.censored_pct.iloc[0] if len(hit) else np.nan
        limits = [row[f'{s}_lod'] for s in ('NASGL', 'NGSA', 'GEMAS')]
        limits = [v for v in limits if pd.notna(v) and v > 0]
        row['lod_ratio_max_min'] = round(max(limits) / min(limits), 3) if len(limits) > 1 else np.nan
        rows.append(row)
    gate = pd.DataFrame(rows)
    gate.to_csv(os.path.join(RESULTS, '00_shared_elements.csv'), index=False)

    a3_pass = len(shared) >= 10
    divergent = int((gate.lod_ratio_max_min >= 2).sum())
    e0a_pass = divergent >= 10

    report = [
        '00_ingest report',
        '',
        'assumptions in force',
        f'  NGSA stratum        {args.ngsa_depth}, {args.ngsa_grainsize}',
        f'  NGSA method order   {args.ngsa_method_order}',
        '  GEMAS digestion     aqua regia (XRF block not used for the shared matrix)',
        '  GEMAS censoring     reconstructed from the LOD/2 signature, NOT read from a flag',
        '',
        'records',
    ]
    for name, (meta, conc, mask, _) in surveys.items():
        report.append(f'  {name.upper():6} {len(meta):>7,} rows  {conc.shape[1]:>3} elements  '
                      f'{100 * mask.to_numpy().mean():>6.2f}% of cells censored')
    report += [
        '',
        f'shared elements across all three surveys: {len(shared)}',
        '  ' + ', '.join(shared) if shared else '  none',
        '',
        f'A3 gate  (>=10 shared elements)              {"PASS" if a3_pass else "FAIL"}',
        f'E0a gate (>=10 shared with LOD ratio >= 2x)  '
        f'{"PASS" if e0a_pass else "FAIL"}   [{divergent} of {len(shared)}]',
        '',
        'GEMAS detection limits are RECOVERED, not published values. Verify every row of',
        '00_gemas_lod_recovery.csv against the published GEMAS limits and set',
        'verified_against_published_lod before any recovered limit is used downstream.',
    ]
    if len(e_recovery):
        report += [
            f'  corroboration: dominance over the next most frequent value ranges from '
            f'{e_recovery.dominance_ratio.min():.0f}x to '
            f'{e_recovery.dominance_ratio.max():.0f}x, and the substituted share from '
            f'{100 * e_recovery.share_at_spike.min():.1f}% to '
            f'{100 * e_recovery.share_at_spike.max():.1f}%.',
            '  Being the column minimum is NOT corroboration here: it is one of the',
            '  acceptance tests, so every recovered spike satisfies it by construction.',
        ]
    report += [
        '',
        'written to 02-datasets/interim/:',
        '  ' + ', '.join(sorted(set(written))),
    ]
    text = '\n'.join(report)
    with open(os.path.join(RESULTS, '00_ingest_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
