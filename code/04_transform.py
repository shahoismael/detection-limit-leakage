"""
04_transform - the pooled composition and the frozen log-ratio geometry

This script does NOT transform the survey data. That is deliberate and it is the
whole point of running it separately.

Log-ratio coordinates are undefined at zero, and every non-detect is a zero until
some handling regime decides what to put there. Transforming first would force a
substitution before the comparison of substitutions has been made, and the paper
would then be measuring its own preprocessing. The operation order is therefore
fixed: a regime fills values in composition space (05_regimes.py), and only then
are coordinates computed from the filled composition, using the geometry frozen
here.

What this script produces is that geometry, plus the pooled matrices every later
stage reads.

  1. The pooled composition: the 44 elements shared by all three surveys, in
     mg/kg, in one fixed element order, with the mask and the per-cell detection
     limit aligned cell for cell.
  2. A sequential binary partition for the isometric log-ratio, chosen from
     geochemical affinity and written to disk so that it cannot drift between
     runs or between experiments.
  3. The isometric log-ratio contrast matrix implied by that partition, written
     out as numbers rather than as code, so that 05, 06 and 07 load one frozen
     matrix instead of re-deriving it and possibly disagreeing.
  4. A validation report that checks the geometry is what it claims to be:
     orthonormal contrasts, isometry against Aitchison distance, centred
     log-ratio rows summing to zero, and an exact round trip.

Why the partition is fixed in advance. The isometric log-ratio is the
pre-registered primary representation because the centred log-ratio is singular,
with residual correlation -1/(D-1) between coordinates. That artefact is
harmless when D is constant and poisonous when it is not: a survey-dependent D
would put a survey-dependent artefact into a study whose subject is
survey-dependent artefacts. D is held at 44 for all three surveys here, and the
report states so explicitly rather than leaving it to be assumed.

Outputs to 02-datasets/processed/
    pooled_meta          survey, sample id, site id, latitude, longitude, stratum
    pooled_conc          concentrations, mg/kg, shared elements, fixed order
    pooled_mask          1 where the provider reported a non-detect
    pooled_lod           the detection limit in force for that cell
    ilr_partition.csv    the sequential binary partition, one row per balance
    ilr_contrast_matrix.csv   the frozen contrast matrix, 43 x 44

Outputs to 05-results/
    04_transform_report.txt

Usage
    python 04_transform.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INTERIM = os.path.join(HERE, '..', '02-datasets', 'interim')
PROCESSED = os.path.join(HERE, '..', '02-datasets', 'processed')
RESULTS = os.path.join(HERE, '..', '05-results')
SURVEYS = ['nasgl', 'ngsa', 'gemas']
SEED = 20260906

# Element order by geochemical affinity. This is the pre-registered ordering and
# it is the only place the sequential binary partition comes from: the partition
# is built by halving this structure, so changing a name here changes the
# coordinate system and invalidates every downstream comparison. The groups are
# Goldschmidt affinity classes, coarsened to what the shared element set can
# actually populate.
AFFINITY_GROUPS = [
    ('rock_forming_lithophile', ['Al', 'Fe', 'Ca', 'Mg', 'Na', 'K', 'Ti', 'Mn', 'P']),
    ('alkali_alkaline_earth', ['Li', 'Be', 'Rb', 'Cs', 'Sr', 'Ba']),
    ('hfse_ree', ['Sc', 'Y', 'La', 'Ce', 'Nb', 'Th', 'U', 'Ga']),
    ('transition_siderophile', ['V', 'Cr', 'Co', 'Ni', 'Cu', 'Zn', 'Mo', 'W']),
    ('chalcophile_volatile', ['Ag', 'As', 'Bi', 'Cd', 'Hg', 'In', 'Pb', 'S',
                              'Sb', 'Se', 'Sn', 'Te', 'Tl']),
]


def read_table(base, folder=INTERIM):
    for extension, reader in (('.parquet', pd.read_parquet), ('.csv.gz', pd.read_csv)):
        path = os.path.join(folder, base + extension)
        if os.path.exists(path):
            return reader(path)
    sys.exit(f'missing input: {base} in {folder}. Run 00_ingest.py first.')


def write_table(frame, base, folder):
    os.makedirs(folder, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        path = os.path.join(folder, base + '.parquet')
        frame.to_parquet(path, index=False)
    except ImportError:
        path = os.path.join(folder, base + '.csv.gz')
        frame.to_csv(path, index=False, compression='gzip')
    return os.path.basename(path)


# ------------------------------------------------------------------ geometry
def build_sequential_binary_partition(groups):
    """Recursive halving, first across affinity groups and then within each one.

    Halving is used rather than a hand-drawn tree because it is deterministic,
    reproduces from the group list alone, and yields exactly D-1 balances with no
    possibility of an accidentally invalid partition. The first balances contrast
    whole affinity classes, which is what makes the leading coordinates readable;
    the later ones resolve individual elements inside a class.
    """
    rows = []

    def split_elements(parts):
        if len(parts) < 2:
            return
        half = len(parts) // 2
        left, right = parts[:half], parts[half:]
        rows.append((left, right))
        split_elements(left)
        split_elements(right)

    def split_groups(subset):
        if len(subset) == 1:
            split_elements(subset[0][1])
            return
        half = len(subset) // 2
        left, right = subset[:half], subset[half:]
        rows.append(([e for _, g in left for e in g],
                     [e for _, g in right for e in g]))
        split_groups(left)
        split_groups(right)

    split_groups(groups)
    return rows


def contrast_matrix(partition, elements):
    """The orthonormal contrast matrix Psi implied by the partition.

    Balance i contrasts r parts against s parts and carries the normalising
    weights +sqrt(s/(r(r+s))) and -sqrt(r/(s(r+s))), which is what makes the
    resulting coordinates orthonormal and the transformation an isometry rather
    than merely a log-ratio.
    """
    position = {element: i for i, element in enumerate(elements)}
    psi = np.zeros((len(partition), len(elements)))
    for i, (plus, minus) in enumerate(partition):
        r, s = len(plus), len(minus)
        for element in plus:
            psi[i, position[element]] = np.sqrt(s / (r * (r + s)))
        for element in minus:
            psi[i, position[element]] = -np.sqrt(r / (s * (r + s)))
    return psi


def centred_log_ratio(values):
    """CLR. Every value must be strictly positive; zeros are a caller error."""
    logged = np.log(np.asarray(values, dtype=float))
    return logged - logged.mean(axis=1, keepdims=True)


def isometric_log_ratio(values, psi):
    return centred_log_ratio(values) @ psi.T


def isometric_log_ratio_inverse(coordinates, psi):
    """Back to a closed composition, for the round-trip check."""
    clr = np.asarray(coordinates, dtype=float) @ psi
    composition = np.exp(clr)
    return composition / composition.sum(axis=1, keepdims=True)


def aitchison_distance(a, b):
    return np.sqrt((((centred_log_ratio(a) - centred_log_ratio(b)) ** 2)).sum(axis=1))


def validate_geometry(psi, elements, seed=SEED):
    """Check the geometry is what it claims, on synthetic strictly positive data.

    Synthetic rather than real, because the real matrix contains non-detects and
    the point here is to test the transformation, not the data. A Dirichlet draw
    gives strictly positive compositions with no structure to flatter the test.
    """
    rng = np.random.default_rng(seed)
    n, D = 400, len(elements)
    sample = rng.dirichlet(np.ones(D), size=n)
    other = rng.dirichlet(np.ones(D), size=n)

    identity_error = float(np.abs(psi @ psi.T - np.eye(psi.shape[0])).max())
    clr_row_sums = float(np.abs(centred_log_ratio(sample).sum(axis=1)).max())

    coordinates = isometric_log_ratio(sample, psi)
    euclidean = np.sqrt(((coordinates - isometric_log_ratio(other, psi)) ** 2).sum(axis=1))
    isometry_error = float(np.abs(euclidean - aitchison_distance(sample, other)).max())

    recovered = isometric_log_ratio_inverse(coordinates, psi)
    round_trip_error = float(np.abs(recovered - sample).max())

    # The singularity the ILR exists to avoid, measured rather than asserted.
    independent = rng.lognormal(size=(2000, D))
    clr_correlation = np.corrcoef(centred_log_ratio(independent), rowvar=False)
    off_diagonal = clr_correlation[~np.eye(D, dtype=bool)]
    return {'contrast_orthonormality_max_error': identity_error,
            'clr_row_sum_max_abs': clr_row_sums,
            'isometry_max_error': isometry_error,
            'round_trip_max_error': round_trip_error,
            'clr_mean_off_diagonal_correlation': float(off_diagonal.mean()),
            'clr_expected_off_diagonal': -1.0 / (D - 1)}


# -------------------------------------------------------------------- pooling
def pool_surveys():
    """One matrix per quantity, surveys stacked, elements in the fixed order."""
    order = [element for _, group in AFFINITY_GROUPS for element in group]

    shared = None
    for survey in SURVEYS:
        columns = set(read_table(f'{survey}_mask').columns)
        shared = columns if shared is None else shared & columns
    shared = sorted(shared)

    missing = sorted(set(shared) - set(order))
    extra = sorted(set(order) - set(shared))
    if missing or extra:
        sys.exit('AFFINITY_GROUPS does not match the shared element set.\n'
                 f'  in the data but not in the ordering: {missing}\n'
                 f'  in the ordering but not in the data: {extra}\n'
                 'Fix the ordering deliberately; do not let it be inferred, or the '
                 'coordinate system changes silently between runs.')

    meta, conc, mask, lod, widths = [], [], [], [], {}
    for survey in SURVEYS:
        m = read_table(f'{survey}_meta')
        c = read_table(f'{survey}_conc')
        k = read_table(f'{survey}_mask')
        d = read_table(f'{survey}_lod')
        widths[survey.upper()] = len(set(c.columns))
        meta.append(m)
        conc.append(c[order].astype(float))
        mask.append(k[order].astype('int8'))
        lod.append(d[order].astype(float))

    return (pd.concat(meta, ignore_index=True),
            pd.concat(conc, ignore_index=True),
            pd.concat(mask, ignore_index=True),
            pd.concat(lod, ignore_index=True),
            order, widths)


def audit_pooled(meta, conc, mask, order):
    """Checks that decide whether the later stages are allowed to trust this."""
    measured = conc.where(mask == 0)
    non_positive = int((measured <= 0).sum().sum())
    per_survey = []
    for survey, rows in meta.groupby('survey', sort=False).groups.items():
        index = list(rows)
        latitude = meta.loc[index, 'latitude']
        longitude = meta.loc[index, 'longitude']
        block = conc.loc[index]
        per_survey.append({
            'survey': survey,
            'n_rows': len(index),
            'n_elements': len(order),
            'censored_pct': round(100 * mask.loc[index].to_numpy().mean(), 2),
            'unavailable_pct': round(100 * block.isna().to_numpy().mean(), 2),
            'coordinates_missing': int(latitude.isna().sum() + longitude.isna().sum()),
            'latitude_range': f'{latitude.min():.2f} to {latitude.max():.2f}',
            'longitude_range': f'{longitude.min():.2f} to {longitude.max():.2f}',
        })
    # Range and presence are separate questions and must not be tested together.
    # An earlier version used .between().all(), which NaN fails, so ten GEMAS rows
    # with no coordinates were reported as out-of-range coordinates. They are
    # missing coordinates, which is a different problem with a different remedy.
    located = meta.has_coordinates == 1
    coordinates_valid = bool(
        meta.loc[located, 'latitude'].between(-90, 90).all()
        and meta.loc[located, 'longitude'].between(-180, 180).all())
    return pd.DataFrame(per_survey), non_positive, coordinates_valid


def main():
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    meta, conc, mask, lod, order, widths = pool_surveys()
    # Carried on the row rather than resolved here, because the answer differs by
    # validation protocol: a sample without coordinates cannot be assigned to a
    # spatial block, but it is perfectly valid under random cross-validation and
    # under leave-one-survey-out. 07_validate.py drops these rows from the
    # spatially blocked arm only, and reports how many it dropped.
    meta['has_coordinates'] = (meta.latitude.notna()
                               & meta.longitude.notna()).astype('int8')
    print(f'pooled {len(conc):,} rows x {len(order)} shared elements')

    partition = build_sequential_binary_partition(AFFINITY_GROUPS)
    psi = contrast_matrix(partition, order)
    if psi.shape != (len(order) - 1, len(order)):
        sys.exit(f'invalid partition: {psi.shape[0]} balances for {len(order)} parts, '
                 f'expected {len(order) - 1}')

    checks = validate_geometry(psi, order)
    survey_table, non_positive, coordinates_valid = audit_pooled(meta, conc, mask, order)

    partition_table = pd.DataFrame([
        {'balance': i + 1,
         'n_plus': len(plus), 'n_minus': len(minus),
         'plus_parts': ' '.join(plus), 'minus_parts': ' '.join(minus)}
        for i, (plus, minus) in enumerate(partition)])
    partition_table.to_csv(os.path.join(PROCESSED, 'ilr_partition.csv'), index=False)
    pd.DataFrame(psi, columns=order,
                 index=[f'balance_{i + 1}' for i in range(psi.shape[0])]) \
        .to_csv(os.path.join(PROCESSED, 'ilr_contrast_matrix.csv'))

    written = [write_table(meta, 'pooled_meta', PROCESSED),
               write_table(conc, 'pooled_conc', PROCESSED),
               write_table(mask, 'pooled_mask', PROCESSED),
               write_table(lod, 'pooled_lod', PROCESSED),
               'ilr_partition.csv', 'ilr_contrast_matrix.csv']

    geometry_ok = (checks['contrast_orthonormality_max_error'] < 1e-10
                   and checks['isometry_max_error'] < 1e-9
                   and checks['round_trip_max_error'] < 1e-12
                   and checks['clr_row_sum_max_abs'] < 1e-12)
    lines = [
        '04_transform report',
        '',
        f'pooled matrix     {len(conc):,} rows x {len(order)} shared elements',
        '  ' + survey_table.to_string(index=False).replace('\n', '\n  '),
        '',
        'ELEMENT ORDER (fixed in advance; the partition is built from it)',
    ]
    for name, group in AFFINITY_GROUPS:
        lines.append(f'  {name:24} {", ".join(group)}')
    lines += [
        '',
        'ISOMETRIC LOG-RATIO GEOMETRY',
        f'  balances                       {psi.shape[0]} for {psi.shape[1]} parts',
        f'  contrast orthonormality error  {checks["contrast_orthonormality_max_error"]:.2e}',
        f'  isometry error vs Aitchison    {checks["isometry_max_error"]:.2e}',
        f'  CLR row sum, max |.|           {checks["clr_row_sum_max_abs"]:.2e}',
        f'  round trip error               {checks["round_trip_max_error"]:.2e}',
        f'  geometry verified              {"yes" if geometry_ok else "NO"}',
        '',
        '  leading balances, which are the interpretable ones',
        '  ' + partition_table.head(4).to_string(index=False).replace('\n', '\n  '),
        '',
        'WHY THE ILR IS PRIMARY',
        f'  CLR coordinates of independent data correlate at '
        f'{checks["clr_mean_off_diagonal_correlation"]:+.5f} on average,',
        f'  against the -1/(D-1) = {checks["clr_expected_off_diagonal"]:+.5f} the '
        f'singularity predicts. The artefact is',
        '  harmless only while D is the same for every survey, which is why D is',
        f'  held at {len(order)} here and stated rather than assumed.',
        '',
        'PREREQUISITES FOR LOG-RATIO WORK',
        f'  non-positive measured values   {non_positive} '
        f'({"OK" if non_positive == 0 else "MUST BE ZERO"})',
        f'  coordinates within valid range {"yes" if coordinates_valid else "NO"} '
        f'(tested on located rows only)',
        f'  rows without coordinates       {int((meta.has_coordinates == 0).sum())} '
        f'- excluded from the spatially blocked arm only, kept for random CV and '
        f'leave-one-survey-out',
        '',
        'ELEMENTS DISCARDED BY THE SHARED-SET RESTRICTION',
        '  ' + ', '.join(f'{s} measured {n}, kept {len(order)}'
                         for s, n in widths.items()),
        '  Restriction is required: if the element set differed by survey, a model',
        '  could separate surveys on which columns exist rather than on which cells',
        '  are censored, and D would differ, reactivating the CLR artefact above.',
        '',
        'NOT DONE HERE, DELIBERATELY',
        '  No survey data was transformed. Non-detects are still empty, and a',
        '  log-ratio is undefined at zero, so transforming now would require a',
        '  substitution before the comparison of substitutions has been made and',
        '  the study would be measuring its own preprocessing. 05_regimes.py fills',
        '  values in composition space; coordinates are computed after that, from',
        '  the contrast matrix frozen here.',
        '',
        'written to 02-datasets/processed/: ' + ', '.join(written),
        'written to 05-results/: 04_transform_report.txt',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, '04_transform_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
