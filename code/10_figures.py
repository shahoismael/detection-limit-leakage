"""
Figures 2 and 3.

Reads  ../05-results/figure_data/figure_2_inflation_vs_lod_ratio.csv
       ../05-results/figure_data/figure_3_protocol_comparison.csv
       ../05-results/08_inflation_vs_lod_ratio.csv  (for the reported rho)
Writes ../06-figs/fig2_inflation_vs_detection_limit_ratio.{pdf,png}
       ../06-figs/fig3_protocol_comparison.{pdf,png}
       ../06-figs/fig2_data.csv, fig3_data.csv   (the plotted values, for captions)

FIGURE 2 is the refutable one. Every element in the pre-registered primary arm is
drawn as its own point against its cross-survey detection-limit ratio, with the
fitted trend over them. Raw points, never binned means: binning would hide the
spread, and the spread is what lets a reader disagree. If the cloud is flat, the
mechanism does not operate on these surveys whatever the simulation showed, and
the figure must let that be seen rather than argued around.

The y-axis is clipped for display only, because a handful of elements collapse to
R2 far below the rest under leave-one-survey-out and would otherwise compress the
region where all the evidence sits. Clipped points are drawn as open triangles at
the boundary with their count stated, so nothing is silently removed - the
alternative, a log or symlog axis, would make an unbounded-below quantity harder
to read rather than easier.

FIGURE 3 shows why spatial blocking is not the remedy. For each arm the three
protocols are drawn on one row, so the eye reads the distance between random and
blocked against the distance between random and leave-one-survey-out. Those two
distances ARE the result: the first is what the recommended fix recovers, the
second is what there was to recover.

Design constraints, identical to Figure 1 so the set reads as one system:
  one measure per panel, never two y-scales
  the same three validated hues in the same fixed order (blue #2a78d6,
    orange #eb6834, aqua #1baf7a), checked for colour-vision deficiency
  identity never by colour alone - every series carries its own marker and a
    direct label, so the figures survive greyscale printing
  recessive grid and axes, no chartjunk, no value printed on every point

Usage
    python 10_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '05-results')
FIGURE_DATA = os.path.join(RESULTS, 'figure_data')
FIGS = os.path.join(HERE, '..', '06-figs')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#d9d8d4'

BLUE = '#2a78d6'      # slot 1
ORANGE = '#eb6834'    # slot 2
AQUA = '#1baf7a'      # slot 3

PRIMARY_REGIME, PRIMARY_REPRESENTATION = 'substitution', 'ilr'
PROTOCOL_STYLE = {
    'random':  (BLUE, 'o', 'pooled random'),
    'blocked': (ORANGE, 's', 'spatially blocked'),
    'loso':    (AQUA, 'D', 'leave-one-survey-out'),
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.edgecolor': INK_SOFT,
    'axes.linewidth': 0.6,
    'xtick.color': INK_SOFT,
    'ytick.color': INK_SOFT,
    'text.color': INK,
    'axes.labelcolor': INK,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})


def read(path):
    if not os.path.exists(path):
        sys.exit(f'missing {path}. Run 09_report.py first.')
    return pd.read_csv(path, float_precision='round_trip')


def tidy(axis):
    axis.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)


def save(figure, stem):
    os.makedirs(FIGS, exist_ok=True)
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(FIGS, f'{stem}.{extension}'))
    plt.close(figure)


def figure_2():
    points = read(os.path.join(FIGURE_DATA,
                               'figure_2_inflation_vs_lod_ratio.csv'))
    primary = points[(points.regime == PRIMARY_REGIME)
                     & (points.representation == PRIMARY_REPRESENTATION)].copy()
    primary = primary[np.isfinite(primary.lod_ratio)
                      & np.isfinite(primary.inflation_random_minus_loso)]
    if primary.empty:
        sys.exit('no points on the primary arm; check 09_report.py output')

    x = primary.lod_ratio.to_numpy(dtype=float)
    y = primary.inflation_random_minus_loso.to_numpy(dtype=float)

    # Display clip only. Chosen from the data so it is not a hand-picked frame:
    # everything inside 1.5 x the interquartile range above the upper quartile is
    # shown at true position, and the rest is marked at the boundary.
    q1, q3 = np.percentile(y, [25, 75])
    ceiling = float(q3 + 1.5 * (q3 - q1))
    ceiling = max(ceiling, float(np.percentile(y, 90)))
    clipped = y > ceiling
    drawn = np.minimum(y, ceiling)

    figure, axis = plt.subplots(figsize=(3.4, 3.0))
    tidy(axis)
    axis.axhline(0, color=INK_SOFT, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    axis.scatter(x[~clipped], drawn[~clipped], s=26, facecolor=BLUE,
                 edgecolor='white', linewidth=0.5, zorder=3, label='element')
    if clipped.any():
        axis.scatter(x[clipped], drawn[clipped], s=30, facecolor='none',
                     edgecolor=BLUE, marker='^', linewidth=0.9, zorder=3)

    logx = np.log10(x)
    slope, intercept = np.polyfit(logx, y, 1)
    grid = np.linspace(logx.min(), logx.max(), 100)
    axis.plot(10 ** grid, slope * grid + intercept, color=ORANGE, linewidth=1.4,
              zorder=4, label='least-squares fit')

    rho = pd.Series(logx).corr(pd.Series(y), method='spearman')
    axis.set_xscale('log')
    axis.set_xlabel('cross-survey detection-limit ratio (×)')
    axis.set_ylabel('inflation:  $R^2_{\\mathrm{random}} - R^2_{\\mathrm{LOSO}}$')
    axis.set_ylim(min(drawn.min(), 0) - 0.08, ceiling + 0.08)
    # Bottom right, which is the quadrant the trend leaves empty: a rising cloud
    # puts its points top-right and bottom-left, so any other corner covers data.
    axis.annotate(f'Spearman $\\rho$ = {rho:+.2f}   n = {len(primary)}'
                  + (f'\n{int(clipped.sum())} above {ceiling:.1f}, drawn at the edge;'
                     f'\ninset shows all points at full range'
                     if clipped.any() else ''),
                  xy=(0.035, 0.975), xycoords='axes fraction',
                  ha='left', va='top', color=INK, fontsize=7.5,
                  linespacing=1.5)

    # Full-range inset. The clipped points are the highest detection-limit ratios
    # in the set, so they are also the highest-leverage observations for the fit.
    # Drawing them only at the boundary leaves a reader unable to judge whether
    # the trend rests on points they cannot see, which is a fair objection to any
    # clipped scatter. The inset costs a corner and answers it.
    if clipped.any():
        inset = axis.inset_axes([0.60, 0.06, 0.36, 0.30])
        inset.scatter(x, y, s=9, facecolor=BLUE, edgecolor='none', alpha=0.75)
        inset.plot(10 ** grid, slope * grid + intercept, color=ORANGE,
                   linewidth=1.0)
        inset.axhline(0, color=INK_SOFT, linewidth=0.5, linestyle=(0, (3, 3)))
        inset.set_xscale('log')
        inset.tick_params(labelsize=5.5, length=2, pad=1)
        inset.set_title('full range', fontsize=6, color=INK_SOFT, pad=2)
        for side in ('top', 'right'):
            inset.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            inset.spines[side].set_color(INK_SOFT)
            inset.spines[side].set_linewidth(0.5)
        inset.set_facecolor(SURFACE)

    # Direct labels on the extremes rather than a legend of 39 element names.
    for _, row in pd.concat([primary.nlargest(2, 'lod_ratio'),
                             primary.nsmallest(1, 'lod_ratio')]).iterrows():
        axis.annotate(row.target,
                      xy=(row.lod_ratio,
                          min(row.inflation_random_minus_loso, ceiling)),
                      xytext=(4, 4), textcoords='offset points',
                      color=INK_SOFT, fontsize=6.8)

    # Above the axes. A scatter with a rising trend has no free corner, and a
    # legend placed in one covers the evidence the figure exists to show.
    axis.legend(handles=[
        Line2D([], [], color=BLUE, marker='o', linestyle='none',
               markeredgecolor='white', markersize=5,
               label=f'element ({PRIMARY_REGIME}, {PRIMARY_REPRESENTATION.upper()})'),
        Line2D([], [], color=BLUE, marker='^', linestyle='none',
               markerfacecolor='none', markersize=5, label='clipped for display'),
        Line2D([], [], color=ORANGE, linewidth=1.4, label='least-squares fit')],
        loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False,
        ncol=2, handletextpad=0.4, columnspacing=1.2)
    save(figure, 'fig2_inflation_vs_detection_limit_ratio')
    primary.to_csv(os.path.join(FIGS, 'fig2_data.csv'), index=False)
    return len(primary), float(rho), int(clipped.sum())


def figure_3():
    arms = read(os.path.join(FIGURE_DATA, 'figure_3_protocol_comparison.csv'))
    arms = arms[arms.model == 'xgb_regression'].copy()
    arms['arm'] = arms.regime + ' / ' + arms.representation.str.upper()

    order = (arms[arms.protocol == 'loso']
             .sort_values('r2_median').arm.tolist())
    positions = {arm: i for i, arm in enumerate(order)}

    figure, axis = plt.subplots(figsize=(4.6, 0.62 * len(order) + 1.2))
    tidy(axis)
    axis.axvline(0, color=INK_SOFT, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)

    for arm in order:
        row = arms[arms.arm == arm]
        y = positions[arm]
        available = row[row.protocol.isin(('random', 'loso'))]
        if len(available) == 2:
            axis.plot(sorted(available.r2_median), [y, y], color=GRID,
                      linewidth=2.4, solid_capstyle='round', zorder=2)
        for protocol, (colour, marker, _) in PROTOCOL_STYLE.items():
            point = row[row.protocol == protocol]
            if point.empty:
                continue
            value = float(point.r2_median.iloc[0])
            axis.plot([float(point.ci_low.iloc[0]), float(point.ci_high.iloc[0])],
                      [y, y], color=colour, linewidth=1.0, alpha=0.55, zorder=3)
            axis.plot(value, y, marker=marker, color=colour, markersize=5.5,
                      markeredgecolor='white', markeredgewidth=0.6, zorder=4)

    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(order)
    axis.set_ylim(-0.6, len(order) - 0.4)
    # Read the element count from the data. It was hardcoded at 44 and would have
    # gone on saying 44 after any change to the rotation.
    counts = sorted(set(arms.n_elements.dropna().astype(int)))
    label = (f'{counts[0]} elements' if len(counts) == 1
             else f'{min(counts)}-{max(counts)} elements')
    axis.set_xlabel(f'$R^2$, median over {label} (95 % interval)')
    # Above the axes, not inside them: on the real data the pooled-random points
    # sit far right and the leave-one-survey-out points far left, so any in-axes
    # corner is occupied by a marker the legend would cover.
    axis.legend(handles=[Line2D([], [], color=colour, marker=marker,
                                linestyle='none', markeredgecolor='white',
                                markersize=5.5, label=name)
                         for colour, marker, name in PROTOCOL_STYLE.values()],
                loc='lower center', bbox_to_anchor=(0.5, 1.01),
                frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.4)
    save(figure, 'fig3_protocol_comparison')
    arms.to_csv(os.path.join(FIGS, 'fig3_data.csv'), index=False)
    return len(order)


def main():
    n_points, rho, n_clipped = figure_2()
    n_arms = figure_3()
    print(f'fig2: {n_points} elements on the primary arm, Spearman rho {rho:+.3f}, '
          f'{n_clipped} clipped for display')
    print(f'fig3: {n_arms} arms x 3 protocols')
    print(f'written to 06-figs/: fig2_inflation_vs_detection_limit_ratio.pdf/.png, '
          f'fig3_protocol_comparison.pdf/.png, fig2_data.csv, fig3_data.csv')


if __name__ == '__main__':
    main()
