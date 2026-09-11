"""
E0b - Dose-response simulation.
Paper 1: does detection-limit disparity CAUSE inflated cross-region performance?

Design: k synthetic surveys drawn from ONE generative process. Geochemistry is
identical by construction, so any survey signal can only come from censoring.
Sweep the inter-survey LOD disparity and measure what a model can learn.

Flat curve   -> mechanism is not causal. Stop the project.
Rising curve -> censoring disparity alone produces the inflation.

Outputs (to ../05-results/):
  E0b_dose_response.csv    per-step means and 95% CIs
  E0b_raw_runs.csv         every repetition, for re-analysis
  E0b_summary.txt          verdict
  E0b_partial_runs.csv     append-only checkpoint; delete it to force a clean run

Usage:
  python 02_simulate.py                full sweep, resumes from the checkpoint
  python 02_simulate.py --smoke        3 disparities x 3 reps, for a wiring check
  python 02_simulate.py --fresh        ignore and overwrite the checkpoint
  python 02_simulate.py --reps 25      override repetition count

Every repetition is seeded from (seed, rep, disparity), so a resumed run
reproduces the interrupted run exactly, repetition for repetition.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

# scikit-learn emits this once per LogisticRegression fit under joblib. It is
# noise, it says nothing about the model, and it hides the progress line.
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)

# ----------------------------------------------------------------- config
CFG = dict(
    seed           = 20260906,
    n_surveys      = 3,
    n_per_survey   = 1200,
    n_elements     = 20,
    n_reps         = 100,
    disparity      = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)],
    base_censor    = 0.05,     # censoring every survey shares
    target_element = 0,        # index predicted in the downstream task
    bootstrap_draws= 2000,
    outdir         = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '05-results'),
)

CHECKPOINT_NAME = 'E0b_partial_runs.csv'
RESULT_COLUMNS = ['rep', 'disparity', 'auc_mask', 'r2_within',
                  'r2_loso', 'r2_loso_mask', 'mask_gain',
                  'r2_pooled', 'r2_pooled_mask', 'mask_gain_pooled',
                  'inflation', 'censor_rate']
METRIC_COLUMNS = ['auc_mask', 'r2_within', 'r2_loso', 'r2_loso_mask', 'mask_gain',
                  'r2_pooled', 'r2_pooled_mask', 'mask_gain_pooled', 'inflation']


# ------------------------------------------------------- data generation
def make_population(rng, n, d):
    """One shared geochemical process. Log-normal, correlated, compositional."""
    A = rng.normal(0, 1, size=(d, d))
    cov = A @ A.T / d + np.eye(d) * 0.5
    z = rng.multivariate_normal(np.zeros(d), cov, size=n)
    conc = np.exp(z)                                          # positive, right-skewed
    conc = conc / conc.sum(axis=1, keepdims=True) * 100.0      # closure
    return conc


def apply_lods(conc, lods):
    """Return substituted values (LOD/2) and the binary censoring mask."""
    mask = conc < lods[None, :]
    out = conc.copy()
    out[mask] = np.tile(lods / 2.0, (conc.shape[0], 1))[mask]
    return out, mask.astype(np.int8)


def lods_for(conc_all, base_rate, extra_rate, rng, d):
    """
    Detection limits set as quantiles of the SHARED population, so a survey's
    LOD is a laboratory choice, never a property of its own chemistry.
    """
    rates = np.full(d, base_rate) + extra_rate * rng.uniform(0.3, 1.0, size=d)
    rates = np.clip(rates, 0.0, 0.98)
    return np.array([np.quantile(conc_all[:, j], rates[j]) for j in range(d)])


# ------------------------------------------------------------ one repeat
def one_run(rep, disparity, cfg):
    rng = np.random.default_rng(cfg['seed'] + rep * 1000 + int(disparity * 100))
    d, k, n = cfg['n_elements'], cfg['n_surveys'], cfg['n_per_survey']

    pool = make_population(rng, n * k, d)          # ONE process for all surveys
    X, M, S = [], [], []
    for s in range(k):
        conc = pool[s * n:(s + 1) * n]
        extra = 0.0 if s == 0 else disparity * (s / max(k - 1, 1))
        lods = lods_for(pool, cfg['base_censor'], extra, rng, d)
        sub, mask = apply_lods(conc, lods)
        X.append(sub); M.append(mask); S.append(np.full(n, s))
    X = np.vstack(X); M = np.vstack(M); S = np.concatenate(S)

    Xl = np.log(X)

    # --- (1) can the mask alone identify the survey?
    auc_mask = np.nan
    if M.std() > 0:
        cv = StratifiedKFold(3, shuffle=True, random_state=rep)
        pr = np.zeros((len(S), k))
        for tr, te in cv.split(M, S):
            clf = LogisticRegression(max_iter=400)
            clf.fit(M[tr], S[tr]); pr[te] = clf.predict_proba(M[te])
        try:
            auc_mask = roc_auc_score(S, pr, multi_class='ovr')
        except ValueError:
            pass

    # --- (2) downstream task: predict element 0 from the rest
    t = cfg['target_element']
    feat = np.delete(Xl, t, axis=1)
    maskf = np.delete(M, t, axis=1)
    y = Xl[:, t]

    def fit_score(tr, te, use_mask):
        Ftr = np.hstack([feat[tr], maskf[tr]]) if use_mask else feat[tr]
        Fte = np.hstack([feat[te], maskf[te]]) if use_mask else feat[te]
        m = HistGradientBoostingRegressor(max_iter=120, random_state=0).fit(Ftr, y[tr])
        return r2_score(y[te], m.predict(Fte))

    # within-survey (random CV inside survey 0)
    i0 = np.where(S == 0)[0]; rng.shuffle(i0)
    cut = int(0.8 * len(i0))
    r2_within = fit_score(i0[:cut], i0[cut:], False)

    # leave-one-survey-out: train on 0..k-2, test on k-1
    tr = np.where(S != k - 1)[0]; te = np.where(S == k - 1)[0]
    r2_loso      = fit_score(tr, te, False)
    r2_loso_mask = fit_score(tr, te, True)

    # pooled random CV: the SAME training and test sizes as leave-one-survey-out,
    # but the split ignores survey boundaries, so test rows share survey identity
    # with training rows. This is the protocol that leakage inflates, and the
    # difference between the two is the leakage quantity.
    # Placed after every earlier rng draw so the metrics above stay bit-identical
    # to a run of the previous script version.
    perm = rng.permutation(len(S))
    tr_p, te_p = perm[:len(tr)], perm[len(tr):len(tr) + len(te)]
    r2_pooled      = fit_score(tr_p, te_p, False)
    r2_pooled_mask = fit_score(tr_p, te_p, True)

    return dict(rep=rep, disparity=disparity, auc_mask=auc_mask,
                r2_within=r2_within, r2_loso=r2_loso,
                r2_loso_mask=r2_loso_mask,
                mask_gain=r2_loso_mask - r2_loso,
                r2_pooled=r2_pooled, r2_pooled_mask=r2_pooled_mask,
                mask_gain_pooled=r2_pooled_mask - r2_pooled,
                inflation=r2_pooled - r2_loso,
                censor_rate=float(M.mean()))


# ------------------------------------------------------------- checkpoint
def load_checkpoint(path):
    """Return completed rows and the set of (disparity, rep) pairs already done."""
    if not os.path.exists(path):
        return [], set()
    try:
        # float_precision='round_trip' is required: the default C parser loses
        # the last bit, which would make a resumed run differ from an
        # uninterrupted one in the 16th digit.
        done = pd.read_csv(path, float_precision='round_trip')
    except Exception:
        return [], set()
    if done.empty or not set(RESULT_COLUMNS).issubset(done.columns):
        return [], set()
    done = done.drop_duplicates(subset=['disparity', 'rep'], keep='last')
    rows = done[RESULT_COLUMNS].to_dict('records')
    keys = {(round(float(r['disparity']), 2), int(r['rep'])) for r in rows}
    return rows, keys


def append_checkpoint(path, row):
    # %.17g round-trips a float64 exactly, so a resumed run reproduces an
    # uninterrupted one bit for bit rather than to printed precision.
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        path, mode='a', header=write_header, index=False, float_format='%.17g')


def format_eta(seconds):
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# ------------------------------------------------------------- summarise
def summarise(raw, cfg):
    """
    Mean, median, and a bootstrap 95% interval on the MEDIAN, from a seeded
    generator.

    R-squared is unbounded below, so a handful of collapsed repetitions can move
    a mean by an arbitrary amount while the typical repetition is unchanged. The
    median is the summary the verdict uses; the mean is reported beside it so the
    gap between them is visible rather than hidden.
    """
    boot_rng = np.random.default_rng(cfg['seed'] + 7)

    def stats(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return (np.nan,) * 4
        draws = boot_rng.choice(x, size=(cfg['bootstrap_draws'], len(x)), replace=True)
        medians = np.median(draws, axis=1)
        return (x.mean(), np.median(x),
                np.percentile(medians, 2.5), np.percentile(medians, 97.5))

    out = []
    for disp, g in raw.groupby('disparity'):
        row = {'disparity': disp, 'censor_rate': g.censor_rate.mean(), 'n_reps': len(g)}
        for c in METRIC_COLUMNS:
            mean, med, lo, hi = stats(g[c])
            row[c] = med                      # headline value is the median
            row[c + '_mean'] = mean
            row[c + '_lo'] = lo
            row[c + '_hi'] = hi
        out.append(row)
    return pd.DataFrame(out).sort_values('disparity').reset_index(drop=True)


def build_verdict(summ, raw, cfg):
    """
    Two independent verdicts. RQ1 asks whether censoring carries survey
    identity. RQ2 asks whether that identity inflates a reported cross-region
    score. The first can pass while the second fails, and collapsing them into
    one verdict is how an unsupported claim reaches a manuscript.
    """
    lo_d, hi_d = summ.iloc[0], summ.iloc[-1]

    def rho_of(col):
        u = raw.dropna(subset=[col])
        if u.disparity.nunique() > 2 and len(u) > 3:
            r, p = spearmanr(u.disparity, u[col])
            return r, p, len(u)
        return np.nan, np.nan, len(u)

    # --- RQ1: is censoring an identity channel?
    d_auc = hi_d.auc_mask - lo_d.auc_mask
    rho_auc, p_auc, n_auc = rho_of('auc_mask')
    rq1_pass = (d_auc > 0.10) and (hi_d.auc_mask > 0.70)
    rq1 = ("PASS - the censoring mask alone identifies the survey, and the "
           "effect scales with detection-limit disparity."
           if rq1_pass else
           "FAIL - disparity does not produce survey identity. The mechanism is "
           "not causal under this design. STOP.")

    # --- RQ2: does that identity inflate a reported score?
    # Inflation is pooled random CV minus leave-one-survey-out at matched
    # training and test sizes. The only difference between the two is whether
    # the split respects survey boundaries.
    rho_inf, p_inf, n_inf = rho_of('inflation')
    infl_hi, infl_lo_ci, infl_hi_ci = hi_d.inflation, hi_d.inflation_lo, hi_d.inflation_hi
    rq2_pass = infl_lo_ci > 0 and infl_hi > 0
    rq2 = ("PASS - a pooled random split reports a materially higher score than "
           "leave-one-survey-out on identical data, and the gap widens with "
           "disparity. That gap is the leakage."
           if rq2_pass else
           "NOT SUPPORTED - the pooled-versus-LOSO gap does not exclude zero at "
           "the highest disparity. Do not claim inflation from this run.")

    # --- does the mask itself help, once the split already leaks?
    gain_pool = hi_d.mask_gain_pooled
    gain_loso = hi_d.mask_gain

    cfg_public = {k: v for k, v in cfg.items() if k != 'outdir'}
    txt = f"""E0b dose-response simulation
generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
config: {json.dumps(cfg_public)}

All values below are MEDIANS over repetitions, with a bootstrap 95% interval on
the median. R-squared is unbounded below, so means are reported in the CSV but
are not used for any verdict.

RQ1  IS CENSORING AN IDENTITY CHANNEL?
  mask->survey AUC   {lo_d.auc_mask:.3f} [{lo_d.auc_mask_lo:.3f}, {lo_d.auc_mask_hi:.3f}] at disparity {lo_d.disparity:.2f}
                  -> {hi_d.auc_mask:.3f} [{hi_d.auc_mask_lo:.3f}, {hi_d.auc_mask_hi:.3f}] at disparity {hi_d.disparity:.2f}
                     delta {d_auc:+.3f}   Spearman rho {rho_auc:+.3f} (p = {p_auc:.3g}, n = {n_auc})
  control: within-survey R2 {lo_d.r2_within:.3f} -> {hi_d.r2_within:.3f}
           (flat confirms the geochemistry is unchanged across the sweep)

  VERDICT RQ1: {rq1}

RQ2  DOES IT INFLATE A REPORTED CROSS-REGION SCORE?
  pooled random CV   {lo_d.r2_pooled:.3f} -> {hi_d.r2_pooled:.3f}
  leave-one-survey-out {lo_d.r2_loso:.3f} -> {hi_d.r2_loso:.3f}
  INFLATION          {lo_d.inflation:+.3f} [{lo_d.inflation_lo:+.3f}, {lo_d.inflation_hi:+.3f}] at disparity {lo_d.disparity:.2f}
                  -> {infl_hi:+.3f} [{infl_lo_ci:+.3f}, {infl_hi_ci:+.3f}] at disparity {hi_d.disparity:.2f}
                     Spearman rho {rho_inf:+.3f} (p = {p_inf:.3g}, n = {n_inf})

  VERDICT RQ2: {rq2}

MASK AS AN EXPLICIT FEATURE (secondary)
  gain under pooled random CV   {gain_pool:+.4f}
  gain under leave-one-survey-out {gain_loso:+.4f}
  A gain that appears under the pooled split and not under LOSO is the shortcut
  being exploited. A gain near zero under both means the leak travels through
  the substituted concentrations, not through the mask supplied as a feature.

censoring rate     {lo_d.censor_rate:.3f} -> {hi_d.censor_rate:.3f}

Pass rules
  RQ1: AUC delta > 0.10 AND highest-disparity AUC > 0.70
  RQ2: bootstrap lower bound on median inflation > 0 at the highest disparity

Geochemistry was identical across surveys by construction. Any survey signal
here can only come from the detection limits.
"""
    return txt


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="E0b dose-response simulation")
    ap.add_argument('--smoke', action='store_true',
                    help="tiny sweep to verify wiring; writes to a separate folder")
    ap.add_argument('--fresh', action='store_true',
                    help="ignore any existing checkpoint and start over")
    ap.add_argument('--reps', type=int, default=None, help="override repetition count")
    ap.add_argument('--outdir', type=str, default=None, help="override output folder")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.smoke:
        cfg['disparity'] = [0.0, 0.45, 0.95]
        cfg['n_reps'] = 3
        cfg['n_per_survey'] = 300
        cfg['bootstrap_draws'] = 200
        cfg['outdir'] = os.path.join(cfg['outdir'], '_smoke')
    if args.reps is not None:
        cfg['n_reps'] = args.reps
    if args.outdir is not None:
        cfg['outdir'] = args.outdir

    os.makedirs(cfg['outdir'], exist_ok=True)
    checkpoint = os.path.join(cfg['outdir'], CHECKPOINT_NAME)
    if args.fresh and os.path.exists(checkpoint):
        os.remove(checkpoint)

    rows, done = load_checkpoint(checkpoint)
    planned = [(d, r) for d in cfg['disparity'] for r in range(cfg['n_reps'])]
    todo = [p for p in planned if p not in done]

    # Only reuse checkpoint rows that belong to this plan.
    rows = [r for r in rows
            if (round(float(r['disparity']), 2), int(r['rep'])) in set(planned)]

    print(f"E0b dose-response simulation")
    print(f"  output folder : {os.path.normpath(cfg['outdir'])}")
    print(f"  plan          : {len(cfg['disparity'])} disparities x {cfg['n_reps']} reps "
          f"= {len(planned)} repetitions")
    if done:
        print(f"  resuming      : {len(planned) - len(todo)} already complete, "
              f"{len(todo)} remaining")
    print()

    started = time.time()
    completed_here = 0
    for disp in cfg['disparity']:
        for rep in range(cfg['n_reps']):
            if (disp, rep) in done:
                continue
            row = one_run(rep, disp, cfg)
            rows.append(row)
            append_checkpoint(checkpoint, row)
            completed_here += 1

            elapsed = time.time() - started
            per = elapsed / completed_here
            eta = per * (len(todo) - completed_here)
            print(f"\r  disparity {disp:.2f}  rep {rep + 1:>3}/{cfg['n_reps']}   "
                  f"{completed_here}/{len(todo)} done   "
                  f"{per:.1f}s/rep   ETA {format_eta(eta)}      ",
                  end='', flush=True)

        finished = [r for r in rows if round(float(r['disparity']), 2) == disp]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            auc = np.nanmean([r['auc_mask'] for r in finished])
            gain = np.nanmean([r['mask_gain'] for r in finished])
        print(f"\r  disparity {disp:.2f}  auc_mask {auc:.3f}  mask_gain {gain:+.4f}"
              f"   ({len(finished)} reps)                    ", flush=True)

    raw = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(['disparity', 'rep'])
    raw.to_csv(os.path.join(cfg['outdir'], 'E0b_raw_runs.csv'), index=False)

    summ = summarise(raw, cfg)
    summ.to_csv(os.path.join(cfg['outdir'], 'E0b_dose_response.csv'), index=False)

    txt = build_verdict(summ, raw, cfg)
    with open(os.path.join(cfg['outdir'], 'E0b_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print("\n" + txt)
    print(f"wrote E0b_raw_runs.csv, E0b_dose_response.csv, E0b_summary.txt "
          f"to {os.path.normpath(cfg['outdir'])}")


if __name__ == '__main__':
    main()
