# E0a - Censoring audit. GATE RESULT.

Run on downloaded data, not on documentation. Three surveys, three different
states of the same problem.

## VERDICT: GATE PASSED. Stronger than the design assumed.

---

## NASGL (USA) - non-detects FLAGGED
Appendix_3b_Ahorizon, n=4,858, 82 columns, 40 carry censored values.
Format: "<value". One LOD per element, constant across the survey.

  Ag  <1      98.8%      Cd  <0.1    23.6%
  Te  <0.1    95.8%      In          20.9%
  Cs  <5      81.1%      S           12.0%
  Se  <0.2    43.6%      Hg  <0.01    7.9%

A1 pass (LOD readable from the flag). A2 pass (flag intact).

---

## NGSA (Australia) - non-detects FLAGGED, LOD IN THE HEADER
Rec2011_020_110706.csv, n=7,890, 146 columns carry censored values.
Column header format: "Ag ICP-MS mg/kg 0.03" - element, method, unit, LOD.

THE KEY STRUCTURAL FIND: the same element is measured by three digestion
methods, each with its own detection limit, on the SAME samples.

  element   ICP-MS    Aqua Regia   MMI-ME     spread
  Ag        0.03      0.002        0.001      30x
  As        0.4       0.1          0.01       40x
  Cd        0.1       0.01         0.001      100x
  Ba        0.5       0.5          0.01       50x

Censoring by method: Ta AR 100%, Ag ICP-MS 98.7%, W AR 87.3%, Ge AR 82.1%.

A1 pass. A2 pass.

CONSEQUENCE FOR THE PAPER: this gives the discordant-censoring test (E1c)
INSIDE ONE SURVEY. Same sample, same laboratory, same day - censored under one
method, quantified under another. Concentration is identical by construction,
so threat T3 (geology-survey confounding), the paper's largest, is eliminated
by design rather than by statistical control.

---

## GEMAS (Europe) - non-detects ALREADY SUBSTITUTED AT LOD/2, FLAG DESTROYED

Ap Aqua Regia + XRF, n=2,113, 141 columns.
No "<" anywhere. Every element field is numeric.

The signature is unambiguous. In every affected element the modal low value is
EXACTLY HALF the second-lowest distinct value:

  element  substituted value  share    next distinct   ratio
  Re       0.00025            76.8%    0.0005          1/2
  Pt       0.0005             69.5%    0.001           1/2
  Pd       0.0005             59.4%    0.001           1/2
  Te       0.01               55.6%    0.02            1/2
  Ge       0.01               25.6%    0.02            1/2
  Au       0.0001             15.4%    0.0002          1/2
  B        0.25                7.4%    0.5             1/2
  In       0.0015              3.8%    0.003           1/2
  Se       0.025               2.2%    0.05            1/2

A2 FAILS. A1 passes by inference - the detection limit is exactly twice the
spike value, recoverable to full precision.

GEMAS IS STILL USABLE. The mask that was destroyed can be reconstructed
exactly: any value equal to the spike constant is a non-detect.

---

## WHY THIS IS THE PAPER

Three continental surveys sit in three different states:
  NASGL   flagged, LOD in the data
  NGSA    flagged, LOD in the header, multi-method by design
  GEMAS   substituted at LOD/2, flag removed, undocumented in the delivered file

Europe's largest and most-cited soil geochemical survey has already performed
the operation this paper argues is harmful, silently, before delivery. Anyone
who downloads GEMAS and models it alongside NASGL or NGSA is training on
substituted values without being told.

Worse, the substituted constants are literal survey tags. The value 0.00025
appears 1,619 times in GEMAS Re and appears nowhere in the other two surveys.
A model does not need to infer laboratory identity from a subtle pattern. It
can read it off a repeated constant.

This is no longer a hypothesis about leakage. It is a documented property of
the three datasets, verified on the delivered files.

## STATUS OF THE ASSUMPTIONS
A1 detection limits recoverable      PASS on all three
A2 non-detects flagged               PASS NASGL, PASS NGSA, FAIL GEMAS
A3 >=10 shared elements              not yet checked
A4 survey as laboratory proxy        holds

## NEXT
1. Extract the shared element list across all three (A3).
2. Reconstruct the GEMAS mask from the spike constants.
3. Run E1a/E1b - mask-only survey classifier.
4. Run E1c on NGSA alone, across its three methods. That result does not
   depend on any cross-survey comparison and cannot be confounded by geology.
