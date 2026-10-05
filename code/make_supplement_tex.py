#!/usr/bin/env python3
"""从结果文件生成补充材料 LaTeX(paper_jne/supplement.tex),与正文同用 iopart。S1–S15;表编号 S1, S2, …(审稿 P6)。"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
T = json.load(open(os.path.join(R, 'tables_v3.json')))
S = T['stats']
C = json.load(open(os.path.join(R, 'p1_causal_replay', 'results_thr100_mu0.05.json')))
H = json.load(open(os.path.join(R, 'p1_clean_eval', 'heldout_param_selection.json')))
HG = json.load(open(os.path.join(R, 'p1_causal_replay', 'heldout_gate_threshold.json')))
IMU = json.load(open(os.path.join(R, 'p1_clean_eval', 'imu_only_decoding.json')))['summary']
W = json.load(open(os.path.join(R, 'wrist_encoder_reference', 'results.json')))['summary']
SW = T['sweeps']
SPEEDS = ['0.0', '0.8', '1.6', '2.0']
LABEL = {'asr_std': 'asr', 'asr10_std': 'asr10', 'cca_ho': 'cca', 'nlms_ho': 'gated NLMS', 'icc_w4': 'icc 4 s', 'gait+reg': 'gait+reg', 'gait': 'gait',
         'reg_b10': 'reg 10 s', 'reg_b30': 'reg 30 s', 'nlms': 'NLMS (ungated)', 'reg': 'reg', 'asr': 'rank-safe asr', 'asr10': 'rank-safe asr10',
         'cca40_nolag': 'global, no lags', 'icc_w4_lag': '4 s, lags', 'icc_w20': '20 s, no lags', 'icc_w20_lag': '20 s, lags',
         'reg_shift10': 'shift 10 s', 'reg_shift': 'shift 30 s', 'reg_shift120': 'shift 120 s', 'reg_surr': 'phase-rnd',
         'nlms_shift10': 'shift 10 s', 'nlms_shift': 'shift 30 s', 'nlms_shift120': 'shift 120 s', 'nlms_surr': 'phase-rnd',
         'reg_head': 'head only', 'reg_ankle': 'ankles only'}


def hs(h):
    return '$<10^{-5}$' if h < 1e-5 else (f'{h:.1g}' if h < 0.001 else f'{h:.3f}' if h < 0.1 else f'{h:.2f}')


def row(r):
    ud = f"{r['up']}/{r['down']}" + (f"/{r['tie']}" if r['tie'] else '')
    return f"${r['mean']:+.2f}$ [${r['lo']:+.2f}$, ${r['hi']:+.2f}$] & ${r['hl']:+.2f}$ [${r['hl_lo']:+.2f}$, ${r['hl_hi']:+.2f}$] & {ud} & {r['p']:.2g} & {hs(r['p_holm'])}"


def lab(m):
    return LABEL.get(m, m).replace('_', '\\_').replace('%', '\\%')


def fam_table(key, caption, label, method_col=True):
    """一个家族一张长表:条件、方法、均值差、HL、升降、p、Holm。"""
    if key not in S:
        return f"% family {key} missing\n"
    rows = S[key]
    head = "Condition & Method & Mean [95\\% CI] & HL [95\\% CI] & Up/down(/tied) & $p$ & Holm\\\\" if method_col else "Condition & Mean [95\\% CI] & HL [95\\% CI] & Up/down(/tied) & $p$ & Holm\\\\"
    cols = '@{}lllllrr' if method_col else '@{}llllrr'
    L = [f"{{\\scriptsize\\setlength{{\\tabcolsep}}{{2.5pt}}\n\\begin{{longtable}}{{{cols}}}\n\\caption{{\\label{{{label}}}{caption}}}\\\\\n\\br\n{head}\n\\mr\n\\endfirsthead\n\\caption[]{{{caption} (continued)}}\\\\\n\\br\n{head}\n\\mr\n\\endhead\n"]
    for r in rows:
        if method_col:
            L.append(f"{r['task']} {r['speed']} ({r['n']}) & {lab(r['b']) if r['a'] in ('none', 'reg', 'reg_b30', 'nlms_ho') and key not in ('F14 rank-safe ASR vs standard-order ASR',) else lab(r['a'])} & {row(r)}\\\\\n")
        else:
            L.append(f"{r['task']} {r['speed']} ({r['n']}) & {row(r)}\\\\\n")
    L.append("\\br\n\\end{longtable}}\n")
    return ''.join(L)


def sweep_table(name, grid_labels, caption):
    cols = 'c' * len(grid_labels)
    fs = '\\scriptsize\\setlength{\\tabcolsep}{2pt}' if len(grid_labels) >= 7 else '\\footnotesize\\setlength{\\tabcolsep}{4pt}'
    L = [f"\\begin{{table}}[h]\n\\caption{{{caption}}}\n\\begin{{indented}}\n\\item[]{fs}\n\\begin{{tabular}}{{@{{}}l{cols}}}\n\\br\nCondition & " + ' & '.join(l.replace('%', '\\%') for _, l in grid_labels) + "\\\\\n\\mr\n"]
    for task in ('ERP', 'SSVEP'):
        for sp in SPEEDS:
            L.append(f"{task} {sp}\\,m/s & " + ' & '.join(SW.get(f'{name}|{task}|{sp}|{m}', '--') for m, _ in grid_labels) + "\\\\\n")
    L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")
    return ''.join(L)


cca_choice = '/'.join(sorted(set(T['heldout']['cca_ho']['fold_choices'])))
L = [r"""\documentclass[12pt]{iopart}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{iopams}
\usepackage{longtable}
\usepackage{url}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}\sloppy\setlength{\tabcolsep}{3pt}
\title[Supplementary material]{Supplementary material for: A decoding-centred benchmark of IMU-referenced motion-artifact removal for ambulatory EEG-BCI, with a causal motion-gated replay}
\author{Yao Xie$^{1,*}$ and Lirui Sun$^{1}$}
\address{$^1$ School of Information Science and Technology, University of Science and Technology of China, Hefei 230026, People's Republic of China}
\ead{xieyao@mail.ustc.edu.cn (Y Xie); kris@mail.ustc.edu.cn (L Sun)}
\address{ORCID iD: Y Xie, \url{https://orcid.org/0000-0001-5649-1513}}
\begin{abstract}
This supplement contains: S1 the device-encoder negative result; S2 inclusion per condition; S3 the full F4 family and parameter sweeps; S4 ASR under three processing orders; S5 held-out parameter selection; S6--S7 surrogate-reference statistics; S8 IMU-only decoding; S9 motion-gate performance per participant and release, and the transition test; S10 the ungated causal replay family; S11 the standing-cost family for all methods; S12 blockwise regression; S13 the CCA window$\times$lag factorial; S14 Single-release (figshare) results; S15 causal front-end sensitivity; S16 cross-release concordance; S17 SSVEP signal probe. Sections are numbered S1--S17 and tables S1, S2, \ldots\ independently; all are generated by \texttt{code/make\_tables.py} and \texttt{code/make\_supplement\_tex.py} from the result files in the public repository.
\end{abstract}
\maketitle
\noindent Differences are mean paired differences across participants with participant-bootstrap percentile 95\% CIs (10\,000 resamples); HL is the Hodges--Lehmann median shift with a bootstrap 95\% CI (2\,000 resamples); ERP differences in AUC $\times 100$ points, SSVEP in percentage points; $p$ from two-sided signed-rank tests with the exact conditional permutation distribution (midranks for ties, zero differences kept and their ranks split), Holm-corrected within the stated family. ``asr'' and ``asr10'' denote standard-order ASR (ASR before interpolation and re-referencing) unless marked rank-safe; ``cca'' is the global lagged CCA with the held-out threshold ($r=""" + cca_choice + r"""$); ``gated NLMS'' is the retrospective gated NLMS with the held-out step size.
"""]

# ---- S1
L.append(r"""\section*{S1. Device encoder as reference: negative result}
Forty-five healthy participants of a public wrist-exoskeleton dataset (8 wet EEG channels over sensorimotor cortex at 512\,Hz; three-degree-of-freedom exoskeleton in transparent, motor-off mode; encoder angles at 100\,Hz; four-direction pointing, 7--9 task runs of 40 ten-second trials per participant; rest runs identified by constant encoder output and excluded). Reference = encoder angle and velocity, 26 lags over $\pm 100$\,ms, ridge regression per channel. Metrics per participant (medians over trials): movement-period (5--7.5\,s) mu and beta power change at C3 relative to the preview period (3--5\,s); across-trial SD of single-trial mu change; broadband (1--40\,Hz) movement/preview power ratio; AUC of a movement-vs-preview detector (C3/C4 mu and beta log power, trial-grouped 5-fold LDA). Trials with any channel exceeding 100\,$\mu$V peak-to-peak were discarded (97.0\% kept).
\begin{table}[h]
\caption{\label{tab:s1}Wrist exoskeleton, 45 participants: before vs after encoder-referenced regression (means over participants; paired difference; improved/worsened; Wilcoxon $p$).}
\begin{indented}
\item[]\footnotesize
\begin{tabular}{@{}p{4.2cm}ccccc}
\br
Metric & none & encoder reg & mean diff & up/down & $p$\\
\mr
""")
for k, labx in [('erd_mu_C3', 'mu change at C3 (\\%)'), ('erd_beta_C3', 'beta change at C3 (\\%)'), ('erd_mu_sd', 'across-trial SD of mu change'), ('broad_ratio', 'broadband power ratio'), ('move_auc', 'movement-detection AUC')]:
    a = W['none_' + k]['mean']; b = W['enc_reg_' + k]['mean']; pd_ = W['paired_' + k]
    L.append(f"{labx} & {a:.3f} & {b:.3f} & ${pd_['mean_diff']:+.3f}$ & {pd_['n_up']}/{pd_['n_down']} & {pd_['p']:.2g}\\\\\n")
L.append(f"\\br\n\\end{{tabular}}\n\\end{{indented}}\n\\end{{table}}\nVariance removed from the EEG by the encoder reference: {W['var_removed_pct']['mean']:.2f}\\% (mean over participants). With the motors off and participants seated, device-synchronous artifact was minimal; the experiment therefore does not test the hypothesis that an actively driven device's own motion signal is a useful reference.\n")

# ---- S2 inclusion
L.append(r"""\section*{S2. Inclusion per condition}
\begin{table}[h]
\caption{\label{tab:s2}Participants included per condition (a participant is included when the unprocessed decoding value exists), the number of those from the figshare release, and participants missing.}
\begin{indented}
\item[]\small
\begin{tabular}{@{}lccp{5.0cm}}
\br
Speed & ERP $n$ (figshare) & SSVEP $n$ (figshare) & Missing (ERP / SSVEP)\\
\mr
""")
for sp in SPEEDS:
    inc = T['inclusion'][sp]
    L.append(f"{sp}\\,m/s & {inc['ERP']} ({inc['ERP_figshare']}) & {inc['SSVEP']} ({inc['SSVEP_figshare']}) & {', '.join(inc['missing_ERP']) or '--'} / {', '.join(inc['missing_SSVEP']) or '--'}\\\\\n")
L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\nReasons: s15 has no standing ERP file; s18 has no SSVEP; s02 has no 0.8\\,m/s SSVEP; figshare participants s15--s18 have no running session, s14 lacks running ERP and s13 lacks running SSVEP; OSF participant sub-16 lacks running SSVEP. The six OSF-only participants all have running sessions.\n")

# ---- S3 F4 full + sweeps
L.append("\\section*{S3. Family F4 (other offline methods vs none) and parameter sweeps}\n")
L.append(fam_table('F4 other offline vs none', 'Family F4, 36 tests (standard-order ASR; held-out CCA threshold): mean [95\\% CI]; HL [95\\% CI]; improved/worsened(/tied); unadjusted and Holm $p$.', 'tab:s3a'))
L.append(sweep_table('cca r', [('none', 'none'), ('cca20', '0.2'), ('cca', '0.3'), ('cca40', '0.4'), ('cca50', '0.5'), ('cca60', '0.6'), ('cca70', '0.7')], 'CCA threshold $r$ (global, lagged): means per condition (ERP AUC; SSVEP \\%).'))
L.append(sweep_table('reg 岭系数', [('none', 'none'), ('reg_a1', '0.01%'), ('reg', '0.1%'), ('reg_a3', '1%')], 'Ridge penalty (reg): means per condition.'))
L.append(sweep_table('NLMS μ', [('none', 'none'), ('nlms_mu02', '0.02'), ('nlms_gated', '0.05'), ('nlms_mu10', '0.10'), ('nlms_mu20', '0.20')], 'Gated NLMS step size $\\mu$: means per condition.'))

# ---- S4 ASR orders
A = T['asr_orders']
L.append(r"""\section*{S4. ASR under three processing orders}
Three orders were run with the same meegkit implementation (Euclidean metric, standing-training-session calibration): (i) CAR and bad-channel interpolation \emph{before} ASR, applied directly; (ii) the same order with the data projected onto the full-rank calibration subspace (``rank-safe''); (iii) the standard order, ASR on the original-reference data with bad channels removed, followed by interpolation and CAR. The main text uses (iii).
\begin{table}[h]
\caption{\label{tab:s4}ERP AUC at 1.6\,m/s under the three orders (cutoff 20) and standard order with cutoff 10.}
\begin{indented}
\item[]\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccc}
\br
Group & $n$ & none & (i) direct & (ii) rank-safe & (iii) standard & (iii) std., cutoff 10\\
\mr
""")
for grp in ('figshare 18', 'OSF 6'):
    a = A[grp]
    dv = '--' if a['direct'] is None else f"{a['direct']:.3f}"
    L.append(f"{grp} & {a['n']} & {a['none']:.3f} & {dv} & {a['ranksafe']:.3f} & {a['std']:.3f} & {a['std10']:.3f}\\\\\n")
er = A.get('energy_ratio', {})
L.append(f"\\br\n\\end{{tabular}}\n\\end{{indented}}\n\\end{{table}}\nOrder (i) was run on the figshare participants and, in the earlier draft, on the six OSF recordings that duplicate figshare participants; on those, with interpolated bad channels, its output energy ranged from 1 to $10^{{18}}$ times the input (one recording: $6.8\\times 10^{{18}}$). It was not re-run on the six OSF-only participants (`--'). Under the standard order (iii) the output/input energy ratio over all {er.get('n_segments', 0)} processed segments was {er.get('min', float('nan')):.3f}--{er.get('max', float('nan')):.3f} (median {er.get('median', float('nan')):.3f}); no divergence occurred. Whether order (i) also diverges in the reference MATLAB implementation was not tested (no MATLAB licence).\n")
L.append(sweep_table('ASR 顺序', [('none', 'none'), ('asr', 'rank-safe 20'), ('asr_std', 'standard 20'), ('asr10', 'rank-safe 10'), ('asr10_std', 'standard 10')], 'Rank-safe (order ii) and standard-order (iii) ASR: means per condition.'))
L.append(fam_table('F14 rank-safe ASR vs standard-order ASR', 'Family F14 (12 tests): rank-safe minus standard-order ASR at the same cutoff.', 'tab:s4b'))

# ---- S5 folds
L.append(r"""\section*{S5. Held-out parameter selection}
""")
L.append(f"Halves (seed 20261004): A = {', '.join(H['halves'][0])}; B = {', '.join(H['halves'][1])}. The selection score is the unweighted mean over the six locomotion conditions of the mean paired change vs none (ERP $\\times 100$).\n\n")
for fam, labx in (('nlms', 'NLMS step size $\\mu$'), ('cca', 'CCA threshold $r$ (grid 0.2--0.7)')):
    h = H[fam]
    L.append(f"\\emph{{{labx}}}: selection scores on A = {{{', '.join(f'{k}: {v:.2f}' for k, v in h['fold1']['scores'].items())}}} (chosen {h['fold1']['chosen']}); on B = {{{', '.join(f'{k}: {v:.2f}' for k, v in h['fold2']['scores'].items())}}} (chosen {h['fold2']['chosen']}); held-out objective {h['heldout_objective']:.2f} points vs in-sample {h['insample_objective']:.2f} (in-sample choice {h['insample_all24']['chosen']}).\n\n")
L.append(f"\\emph{{Gate threshold $\\theta$}} (balanced accuracy of per-second decisions): fold 1 scores {{{', '.join(f'{k}: {v:.3f}' for k, v in HG['fold1']['scores_on_selection_half'].items())}}} (chosen {HG['fold1']['chosen']}; held-out balanced accuracy {HG['fold1']['heldout_balanced_acc']:.3f}, sensitivity {HG['fold1']['heldout_sens']:.3f}, specificity {HG['fold1']['heldout_spec']:.3f}); fold 2 scores {{{', '.join(f'{k}: {v:.3f}' for k, v in HG['fold2']['scores_on_selection_half'].items())}}} (chosen {HG['fold2']['chosen']}; held-out {HG['fold2']['heldout_balanced_acc']:.3f}, {HG['fold2']['heldout_sens']:.3f}, {HG['fold2']['heldout_spec']:.3f}).\n")

# ---- S6/S7 surrogate stats
L.append("\\section*{S6. Surrogate references vs true reference, regression (F6)}\n")
L.append(fam_table('F6 controls vs true reference (reg)', 'Surrogate minus true reference, regression: 24 tests.', 'tab:s6'))
L.append("\\section*{S7. Surrogate references vs true reference, gated NLMS (F7)}\n")
L.append(fam_table('F7 controls vs true reference (NLMS)', 'Surrogate minus true reference, gated NLMS: 24 tests.', 'tab:s7'))

# ---- S8 IMU-only
L.append(r"""\section*{S8. IMU-only decoding}
\begin{table}[h]
\caption{\label{tab:s8}Decoders applied to IMU channels instead of EEG (mean $\pm$ SD over participants). ERP: AUC with the standing-session classifier (tr) and with within-speed 5-fold cross-validation (cv); SSVEP: CCA accuracy (\%, chance 33.3).}
\begin{indented}
\item[]\small
\begin{tabular}{@{}lccc}
\br
Speed & ERP AUC (tr) & ERP AUC (cv) & SSVEP (\%)\\
\mr
""")
for sp in SPEEDS:
    g = lambda k: f"{IMU[k]['mean']:.3f} $\\pm$ {IMU[k]['sd']:.3f} ({IMU[k]['n']})" if k in IMU else '--'
    gs = lambda k: f"{IMU[k]['mean']:.1f} $\\pm$ {IMU[k]['sd']:.1f} ({IMU[k]['n']})" if k in IMU else '--'
    L.append(f"{sp}\\,m/s & {g(f'ERP_{sp}_imu_tr')} & {g(f'ERP_{sp}_imu_cv')} & {gs(f'SSVEP_{sp}_imu')}\\\\\n")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('mt', os.path.join(ROOT, 'code', 'make_tables_lib.py')); _mt = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mt); exact_signed_rank_p = _mt.exact_signed_rank_p
IMUP = json.load(open(os.path.join(R, 'p1_clean_eval', 'imu_only_decoding.json')))['per_subject']
pv = []
for sp in SPEEDS:
    for k, ch, labx in ((f'ERP_{sp}_imu_cv', 0.5, 'ERP cv'), (f'ERP_{sp}_imu_tr', 0.5, 'ERP tr'), (f'SSVEP_{sp}_imu', 100 / 3, 'SSVEP')):
        v = np.array([d[k] for d in IMUP.values() if k in d])
        if len(v):
            pv.append(f"{labx} {sp}\\,m/s $p={exact_signed_rank_p(v - ch):.2g}$")
L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\nExact conditional signed-rank tests against chance (0.5 for AUC, 33.3\\% for SSVEP): " + '; '.join(pv) + ".\n")

# ---- S9 gate per participant + release + transition
per = {}
for seg in C['gate_segments']:
    per.setdefault(seg['sub'], {'m': [], 's': []})
    (per[seg['sub']]['m'] if seg['moving'] else per[seg['sub']]['s']).append(seg['gate_on_frac'])
L.append(r"""\section*{S9. Motion gate per participant and release, and the transition test}
\begin{table}[h]
\caption{\label{tab:s9}Causal gate ($\theta=100$): fraction of per-second decisions with the gate open, averaged over each participant's locomotion segments and standing segments. Participants sub-01, 02, 03, 04, 15 and 16 are from the OSF release (integer-quantized IMU).}
\begin{indented}
\item[]\small
\begin{tabular}{@{}lcc@{\hspace{2em}}lcc}
\br
Participant & locomotion & standing & Participant & locomotion & standing\\
\mr
""")
ks = sorted(per)
half = (len(ks) + 1) // 2
for i in range(half):
    a = ks[i]; b = ks[i + half] if i + half < len(ks) else None
    fa = lambda s: (f"{np.mean(per[s]['m']):.2f}" if per[s]['m'] else '--', f"{np.mean(per[s]['s']):.2f}" if per[s]['s'] else '--')
    ma, sa = fa(a)
    if b:
        mb, sb = fa(b); L.append(f"{a} & {ma} & {sa} & {b} & {mb} & {sb}\\\\\n")
    else:
        L.append(f"{a} & {ma} & {sa} & & & \\\\\n")
g = C['gate']['per_second_decisions']
ts = C['transition_summary']
tw = T['causal']['transition_walk_paired']
rel = T['causal']['gate_by_release']
L.append(f"\\br\n\\end{{tabular}}\n\\end{{indented}}\n\\end{{table}}\nPooled per-second decisions against the segment speed label: TP {g['TP']:.0f}, FN {g['FN']:.0f}, FP {g['FP']:.0f}, TN {g['TN']:.0f}; sensitivity {C['gate']['sensitivity']:.3f}, specificity {C['gate']['specificity']:.3f}. By release: figshare standing segments gate-open fraction mean {rel['figshare']['standing_on_mean']:.3f} (max {rel['figshare']['standing_on_max']:.2f}, {rel['figshare']['n_standing_segments']} segments), locomotion mean {rel['figshare']['moving_on_mean']:.3f} (min {rel['figshare']['moving_on_min']:.2f}); OSF standing mean {rel['OSF']['standing_on_mean']:.3f} (max {rel['OSF']['standing_on_max']:.2f}, {rel['OSF']['n_standing_segments']} segments), locomotion mean {rel['OSF']['moving_on_mean']:.3f} (min {rel['OSF']['moving_on_min']:.2f}). Transition test (standing and 1.6\\,m/s SSVEP recordings concatenated, $n={ts['switch_delay_s']['n']}$): gate switch delay {ts['switch_delay_s']['mean']:.2f}\\,s (SD {ts['switch_delay_s']['sd']:.2f}); gate-open fraction while standing {ts['gate_on_frac_standing']['mean']:.3f}, while walking {ts['gate_on_frac_walking']['mean']:.3f}; standing samples altered {ts['standing_altered_frac']['mean']*100:.1f}\\%; SSVEP accuracy standing {ts['ssvep_standing_none']['mean']:.1f}\\% $\\rightarrow$ {ts['ssvep_standing_gated']['mean']:.1f}\\%, walking {ts['ssvep_walking_none']['mean']:.1f}\\% $\\rightarrow$ {ts['ssvep_walking_gated']['mean']:.1f}\\% (paired change ${tw['mean']:+.2f}$ [${tw['lo']:+.2f}$, ${tw['hi']:+.2f}$], {tw['up']}/{tw['down']}/{tw['tie']}, $p={tw['p']:.2g}$).\n")

# ---- S10 F10
L.append(r"""\section*{S10. Causal replay, ungated NLMS vs none (F10)}
\begin{table}[h]
\caption{\label{tab:s10}Ungated causal NLMS minus none: mean [95\% CI]; HL [95\% CI]; improved/worsened(/tied); unadjusted and Holm $p$ (8 tests).}
\begin{indented}
\item[]\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lllllr}
\br
Condition & Mean [95\% CI] & HL [95\% CI] & up/down(/tied) & $p$ & Holm\\
\mr
""")
for v in T['causal']['rows']:
    u = v['nlms_c']
    ud = f"{u['up']}/{u['down']}" + (f"/{u['tie']}" if u['tie'] else '')
    L.append(f"{u['task']} {u['speed']} ({u['n']}) & ${u['mean']:+.2f}$ [${u['lo']:+.2f}$, ${u['hi']:+.2f}$] & ${u['hl']:+.2f}$ [${u['hl_lo']:+.2f}$, ${u['hl_hi']:+.2f}$] & {ud} & {u['p']:.2g} & {hs(u['p_holm'])}\\\\\n")
L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")

# ---- S11 standing cost all methods
L.append("\\section*{S11. Standing cost for all methods (F5)}\nEvery method applied ungated to the standing ERP and SSVEP recordings, where no locomotion artifact is present. The retrospective gated NLMS leaves standing data untouched by construction (segment-level gate = speed label) and is therefore not part of the family; its causal counterpart is tested in F9.\n")
L.append(fam_table('F5 standing cost', 'Family F5 (14 tests): method minus none on standing data.', 'tab:s11'))

# ---- S12 blockwise regression
L.append("\\section*{S12. Blockwise regression (F11, F12)}\nLagged ridge regression refitted independently in non-overlapping 10-s or 30-s blocks (same lags and relative penalty), as an intermediate between the whole-segment fit and the sample-wise adaptive filter.\n")
L.append(sweep_table('reg 分块', [('none', 'none'), ('reg_b10', '10 s'), ('reg_b30', '30 s'), ('reg', 'whole segment'), ('nlms_ho', 'gated NLMS')], 'Blockwise regression: means per condition.'))
L.append(fam_table('F11 blockwise reg vs reg', 'Family F11 (12 tests): blockwise minus whole-segment regression.', 'tab:s12a'))
L.append(fam_table('F12 gated NLMS vs reg 30 s blocks', 'Family F12 (6 tests): gated NLMS minus 30-s blockwise regression.', 'tab:s12b', method_col=False))

# ---- S13 CCA 2x2
L.append("\\section*{S13. CCA window $\\times$ lag factorial (F13)}\nThe global lagged CCA and the iCanClean-style windowed CCA differ in two factors (whole-segment vs 4-s statistics window; $\\pm 100$\\,ms reference lags vs none) and in their removal criterion ($r>0.4$ held-out vs $R^2>0.85$ default). The two missing cells and a 20-s window were run. With lags, a 4-s window has 378 regressors for 400 samples and is ill-posed (near-unit canonical correlations; almost all EEG energy removed).\n")
L.append(sweep_table('icc 窗 × 时滞', [('none', 'none'), ('cca40', 'global, lags'), ('cca40_nolag', 'global, no lags'), ('icc_w4', '4 s, no lags'), ('icc_w4_lag', '4 s, lags'), ('icc_w20', '20 s, no lags'), ('icc_w20_lag', '20 s, lags')], 'Window $\\times$ lag cells (global cells use $r>0.4$; windowed cells use $R^2>0.85$): means per condition.'))
L.append(fam_table('F13 CCA 2x2 variants vs none', 'Family F13 (24 tests): factorial cells minus none.', 'tab:s13'))

# ---- S14 figshare only
L.append("\\section*{S14. Single-release (figshare) results}\nThe figshare release (18 participants, 500\\,Hz raw, no bad-channel annotations, continuous-valued IMU) analysed alone, as a single-release sensitivity analysis. Thirteen of the 19 ERP runners and 13 of the 18 SSVEP runners of the main analysis are figshare participants; the six OSF-only participants are excluded here.\n")
L.append(fam_table('F1f reg vs none (figshare only)', 'F1 restricted to figshare participants.', 'tab:s14a', method_col=False))
L.append(fam_table('F2f gated NLMS vs none (figshare only)', 'F2 restricted to figshare participants.', 'tab:s14b', method_col=False))
L.append(fam_table('F4f asr10 / cca vs none (figshare only)', 'asr10 and cca vs none restricted to figshare participants.', 'tab:s14c'))
L.append("\\begin{table}[h]\n\\caption{\\label{tab:s14d}Causal replay restricted to figshare participants: gated NLMS minus none (Holm within the 8-test family).}\n\\begin{indented}\n\\item[]\\footnotesize\n\\begin{tabular}{@{}lccllr}\n\\br\nCondition & none & gated & Mean [95\\% CI] & up/down(/tied) & Holm\\\\\n\\mr\n")
for v in T['causal']['rows_figshare']:
    u = v['nlms_gated_c']
    f = (lambda x: f'{x:.3f}') if v['task'] == 'ERP' else (lambda x: f'{x:.1f}')
    ud = f"{u['up']}/{u['down']}" + (f"/{u['tie']}" if u['tie'] else '')
    L.append(f"{v['task']} {v['speed']} ({v['n']}) & {f(v['none_mean'])} & {f(u['method_mean'])} & ${u['mean']:+.2f}$ [${u['lo']:+.2f}$, ${u['hi']:+.2f}$] & {ud} & {hs(u['p_holm'])}\\\\\n")
L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")

# ---- S15 causal front-end sensitivity
if 'causal_hp01' in T:
    fe = T['causal_hp01']['frontend_diff']
    zp = T['causal_vs_zerophase']
    L.append("\\section*{S15. Causal front-end sensitivity}\nThe causal replay repeated with a 0.1\\,Hz second-order causal Butterworth high-pass instead of 0.5\\,Hz (everything else identical, including the decoder trained on the training session processed by the same front end).\n")
    L.append("\\begin{table}[h]\n\\caption{\\label{tab:s15}Causal replay with the 0.1\\,Hz front end: means and gated minus none (Holm within the 8-test family).}\n\\begin{indented}\n\\item[]\\footnotesize\n\\begin{tabular}{@{}lccclr}\n\\br\nCondition & none & ungated & gated & gated $-$ none; up/down(/tied) & Holm\\\\\n\\mr\n")
    for v in T['causal_hp01']['rows']:
        u = v['nlms_gated_c']
        f = (lambda x: f'{x:.3f}') if v['task'] == 'ERP' else (lambda x: f'{x:.1f}')
        ud = f"{u['up']}/{u['down']}" + (f"/{u['tie']}" if u['tie'] else '')
        L.append(f"{v['task']} {v['speed']} ({v['n']}) & {f(v['none_mean'])} & {f(v['nlms_c']['method_mean'])} & {f(u['method_mean'])} & ${u['mean']:+.2f}$ [${u['lo']:+.2f}$, ${u['hi']:+.2f}$]; {ud} & {hs(u['p_holm'])}\\\\\n")
    L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")
    L.append("Unprocessed decoding, paired differences between front ends: 0.1\\,Hz minus 0.5\\,Hz causal: " + '; '.join(f"{x['task']} {x['speed']}: ${x['mean']:+.2f}$ ({x['up']}/{x['down']})" for x in fe) + ". Causal 0.5\\,Hz minus zero-phase retrospective front end: " + '; '.join(f"{x['task']} {x['speed']}: ${x['mean']:+.2f}$ ({x['up']}/{x['down']})" for x in zp) + ".\n")

# ---- S16 cross-release concordance
if 'release_concordance' in T:
    L.append("\\section*{S16. Cross-release concordance}\nThe two public releases contain the same recordings under different participant numbers (matched by stimulus-onset sequences; figshare s01--s18 = OSF sub-11, 12, 05, 13, 14, 06, 19, 07, 08, 17, 09, 10, 20, 18, 21, 22, 23, 24). Six recordings were processed through both releases in an earlier draft (figshare 500\\,Hz raw vs OSF 100\\,Hz with bad-channel interpolation); their decoding results are compared here. Differences are OSF minus figshare for the same recording (ERP in AUC $\\times 100$, SSVEP in points); the effect difference is the method-minus-none change under OSF minus that under figshare.\n")
    L.append("\\begin{table}[h]\n\\caption{\\label{tab:s16}Same recording, two releases: paired differences over the six duplicated recordings.}\n\\begin{indented}\n\\item[]\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lllcrrrr}\n\\br\nMethod & Task & Speed & $n$ & Mean diff & Max $|$diff$|$ & Effect diff mean & Effect diff max\\\\\n\\mr\n")
    for r in T['release_concordance']:
        if r['method'] not in ('none', 'reg', 'nlms_gated', 'asr10_std'):
            continue
        ed = '--' if r['effect_diff_mean'] is None else f"${r['effect_diff_mean']:+.2f}$"
        em = '--' if r['effect_max_abs'] is None else f"{r['effect_max_abs']:.2f}"
        L.append(f"{lab(r['method'])} & {r['task']} & {r['speed']} & {r['n']} & ${r['osf_minus_figshare_mean']:+.2f}$ & {r['max_abs_diff']:.2f} & {ed} & {em}\\\\\n")
    L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")
# ---- S17 SSVEP signal probe
if 'ssvep_probe' in T:
    P = T['ssvep_probe']['summary']; Q = T['ssvep_probe']['paired']
    L.append("\\section*{S17. SSVEP signal probe}\nPer 5-s trial, the CCA correlation with the reference of the attended frequency (``true'') and the largest correlation with the two other references (``wrong''); margin = true $-$ wrong; means over trials then participants. A loss of accuracy with a preserved true correlation would indicate interference at the other frequencies; a loss of the true correlation indicates a weaker or less stable response.\n")
    L.append("\\begin{table}[h]\n\\caption{\\label{tab:s17}SSVEP CCA correlations by speed (means over participants).}\n\\begin{indented}\n\\item[]\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lcccccccc}\n\\br\nSpeed & $n$ & none true & none wrong & none margin & reg true & reg margin & NLMS true & NLMS margin\\\\\n\\mr\n")
    for sp in SPEEDS:
        g = lambda k: f"{P[k]['mean']:.3f}" if k in P else '--'
        n = P.get(f'{sp}_none_margin', {}).get('n', 0)
        L.append(f"{sp}\\,m/s & {n} & {g(f'{sp}_none_corr_true')} & {g(f'{sp}_none_corr_wrong')} & {g(f'{sp}_none_margin')} & {g(f'{sp}_reg_corr_true')} & {g(f'{sp}_reg_margin')} & {g(f'{sp}_nlms_gated_corr_true')} & {g(f'{sp}_nlms_gated_margin')}\\\\\n")
    L.append("\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\nPaired change vs standing (unprocessed): " + '; '.join(f"{k.replace('_vs_standing_', ' vs standing, ').replace('_', ' ')}: ${v['mean']:+.3f}$ ({v['up']}/{v['down']})" for k, v in Q.items()) + ".\n")
L.append("\\end{document}\n")
open(os.path.join(ROOT, 'paper_jne', 'supplement.tex'), 'w', encoding='utf-8').write(''.join(L))
print('supplement.tex written')
