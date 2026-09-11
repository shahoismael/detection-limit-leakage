"""
Figure 1 - E0b dose-response.

Reads  ../05-results/E0b_dose_response.csv
Writes ../06-figs/fig1_dose_response.pdf   (vector, for submission)
       ../06-figs/fig1_dose_response.png   (300 dpi, for drafts)
       ../06-figs/fig1_dose_response_data.csv  (the plotted values, for the caption)

Panel (a) one quantity: how well the censoring mask alone identifies the survey.
Panel (b) one quantity: R-squared under three validation protocols. The shaded
band between the pooled random split and leave-one-survey-out is the inflation -
the leakage is drawn as the area between two protocols, not as a separate line.

Design constraints applied:
  one measure per panel, never two y-scales
  categorical hues assigned in fixed order and validated for colour-vision
    deficiency (blue #2a78d6, orange #eb6834, aqua #1baf7a - worst adjacent pair
    dE 9.2 deutan, 27.6 normal, all checks pass on a light surface)
  identity is never colour alone: every series carries its own line style,
    marker, and a direct label, so the figure survives greyscale printing
  recessive grid and axes; no chartjunk; no value printed on every point
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', '05-results')
FIGS = os.path.join(HERE, '..', '06-figs')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#d9d8d4'

BLUE = '#2a78d6'      # slot 1
ORANGE = '#eb6834'    # slot 2
AQUA = '#1baf7a'      # slot 3

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
    'pdf.fonttype': 42,          # embed TrueType, not Type 3 - required by most journals
    'ps.fonttype': 42,
})


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_SOFT)


def main():
    d = pd.read_csv(os.path.join(RESULTS, 'E0b_dose_response.csv')).sort_values('disparity')
    os.makedirs(FIGS, exist_ok=True)
    x = d.disparity

    # 180 mm wide: full text width in a two-column journal.
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.09, 5.51), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.16})

    # ---------------------------------------------------------- panel (a) AUC
    ax_a.fill_between(x, d.auc_mask_lo, d.auc_mask_hi, color=BLUE, alpha=0.18, linewidth=0)
    ax_a.plot(x, d.auc_mask, color=BLUE, linewidth=2, marker='o', markersize=3.6,
              markerfacecolor=SURFACE, markeredgewidth=1.1, zorder=3)
    ax_a.axhline(0.5, color=INK_SOFT, linewidth=1, linestyle=(0, (1, 2)), zorder=2)
    ax_a.annotate('chance', xy=(0.86, 0.5), xytext=(0.86, 0.515),
                  color=INK_SOFT, fontsize=7)
    ax_a.annotate('mask → survey AUC', xy=(0.62, d.auc_mask.iloc[12]),
                  xytext=(0.40, 0.70), color=BLUE, fontsize=8, fontweight='bold')

    ax_a.set_ylim(0.46, 1.03)
    ax_a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_ylabel('AUC (one-vs-rest)')
    ax_a.set_title('(a)  Censoring alone identifies the survey', loc='left',
                   fontweight='bold', pad=6)
    style_axes(ax_a)

    # ------------------------------------------------- panel (b) R² protocols
    # The gap between the two cross-survey protocols is the inflation.
    ax_b.fill_between(x, d.r2_loso, d.r2_pooled, where=(d.r2_pooled >= d.r2_loso),
                      color=ORANGE, alpha=0.20, linewidth=0, zorder=1,
                      label='inflation (pooled − LOSO)')

    for col, lo, hi, colour, style, marker, label in [
        ('r2_within', 'r2_within_lo', 'r2_within_hi', AQUA, (0, (1, 1.6)), 's',
         'within-survey (control)'),
        ('r2_pooled', 'r2_pooled_lo', 'r2_pooled_hi', BLUE, '-', 'o',
         'pooled random split'),
        ('r2_loso', 'r2_loso_lo', 'r2_loso_hi', ORANGE, (0, (5, 2)), '^',
         'leave-one-survey-out'),
    ]:
        ax_b.fill_between(x, d[lo], d[hi], color=colour, alpha=0.16, linewidth=0)
        ax_b.plot(x, d[col], color=colour, linewidth=2, linestyle=style,
                  marker=marker, markersize=3.6, markerfacecolor=SURFACE,
                  markeredgewidth=1.1, label=label, zorder=3)

    ax_b.axhline(0, color=INK_SOFT, linewidth=0.8, zorder=2)

    # Direct labels: identity is never colour alone.
    ax_b.annotate('pooled random split', xy=(0.60, d.r2_pooled.iloc[12]),
                  xytext=(0.30, 0.47), color=BLUE, fontsize=8, fontweight='bold')
    ax_b.annotate('within-survey (control)', xy=(0.60, d.r2_within.iloc[12]),
                  xytext=(0.30, -0.02), color=AQUA, fontsize=8, fontweight='bold')
    ax_b.annotate('leave-one-survey-out', xy=(0.905, d.r2_loso.iloc[18]),
                  xytext=(0.50, -0.62), color=ORANGE, fontsize=8, fontweight='bold',
                  ha='left',
                  arrowprops=dict(arrowstyle='-', color=ORANGE, linewidth=0.8,
                                  shrinkA=3, shrinkB=3))

    ax_b.set_ylim(-2.20, 0.62)
    ax_b.set_xlim(-0.02, 0.97)
    ax_b.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
    ax_b.set_xlabel('inter-survey detection-limit disparity')
    ax_b.set_ylabel('$R^2$ on the held-out set')
    ax_b.set_title('(b)  The pooled split is blind to the failure it reports on',
                   loc='left', fontweight='bold', pad=6)
    style_axes(ax_b)

    leg = ax_b.legend(loc='lower left', frameon=True, framealpha=0.95,
                      edgecolor=GRID, facecolor=SURFACE, borderpad=0.45,
                      handlelength=2.2, labelspacing=0.35, ncol=2,
                      columnspacing=1.2)
    leg.get_frame().set_linewidth(0.5)
    for text in leg.get_texts():
        text.set_color(INK)

    fig.align_ylabels([ax_a, ax_b])
    fig.savefig(os.path.join(FIGS, 'fig1_dose_response.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'fig1_dose_response.png'), dpi=300, bbox_inches='tight')

    cols = ['disparity', 'censor_rate', 'auc_mask', 'auc_mask_lo', 'auc_mask_hi',
            'r2_within', 'r2_pooled', 'r2_loso', 'inflation',
            'inflation_lo', 'inflation_hi']
    d[cols].round(4).to_csv(
        os.path.join(FIGS, 'fig1_dose_response_data.csv'), index=False)

    lo, hi = d.iloc[0], d.iloc[-1]
    print('wrote fig1_dose_response.pdf / .png / _data.csv to',
          os.path.normpath(FIGS))
    print(f'  AUC       {lo.auc_mask:.3f} -> {hi.auc_mask:.3f}')
    print(f'  pooled    {lo.r2_pooled:.3f} -> {hi.r2_pooled:.3f}')
    print(f'  LOSO      {lo.r2_loso:.3f} -> {hi.r2_loso:.3f}')
    print(f'  inflation {lo.inflation:+.3f} -> {hi.inflation:+.3f}')


if __name__ == '__main__':
    main()
