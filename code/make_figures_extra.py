#!/usr/bin/env python3
"""论文一 · 修订新增四张图(PDF+PNG → paper_jne/figs/):
  fig_protocol   评价协议示意图(数据/映射、前端、方法、解码器、留人、家族、对照、因果回放)
  fig_gate       因果门控时序:拼接的站立→快走段上运动能量比、门控状态、Oz 处理前后
  fig_forest     森林图:F1/F2/F4/F5/F11 的配对差与 95% CI(ERP 与 SSVEP 两栏)
  fig_probe_topo SSVEP 探针(正确/竞争频率相关随速度)+ 回归去除方差的头皮分布
用法: make_figures_extra.py [并行数]
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper_jne', 'figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.unicode_minus': False})
T = json.load(open(os.path.join(ROOT, 'results', 'tables_v3.json')))
S = T['stats']
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FIGS = os.environ.get('FIGS', 'protocol,forest,gate,probe').split(',')


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200)
    plt.close(fig)


# ---------------- fig_protocol ----------------
def box(ax, x, y, w, h, text, fc='#f3f6fa', ec='#345', fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#345'))


if 'protocol' in FIGS:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    FS = 6.6
    box(ax, 0.2, 8.75, 9.6, 1.05, 'Public dataset (Lee et al. 2021): 24 participants; 0 / 0.8 / 1.6 / 2.0 m/s; ERP and SSVEP tasks\n'
        'figshare release s01-s18 (500 Hz raw) + OSF release sub-01/02/03/04/15/16 (100 Hz)\n'
        'participant identity reconciled by stimulus-onset-sequence matching (six recordings in both releases: S16)', fs=FS)
    box(ax, 0.2, 7.2, 4.7, 1.25, 'Retrospective front end\nzero-phase 0.5 Hz high-pass, CAR, 100 Hz\nIMU 18 channels (acc + gyro), per-segment z-score', fs=FS)
    box(ax, 5.1, 7.2, 4.7, 1.25, 'Causal front end (only past samples)\ncausal 0.5 Hz high-pass (0.1 Hz: S15), CAR, 40 Hz low-pass\nIMU zero-order hold, running normalization', fs=FS)
    arrow(ax, 2.55, 8.75, 2.55, 8.45); arrow(ax, 7.45, 8.75, 7.45, 8.45)
    box(ax, 0.2, 5.1, 4.7, 1.8, 'Adapted methods (fitted on the test recording, no labels)\n'
        'reg: lagged IMU ridge (whole segment; 10 s / 30 s blocks)\n'
        'cca: global CCA with lags (no-lag cell: S13)\n'
        'icc: windowed CCA, R2 > 0.85  |  gait, gait+reg\n'
        'asr: standard order (rank-safe variant: S4)\n'
        'NLMS; gated NLMS (segment gate = speed label)', fs=FS)
    box(ax, 5.1, 5.1, 4.7, 1.8, 'Causal replay\n'
        'NLMS with 100 ms EEG delay\n'
        'motion gate: trailing 2-s head-acceleration energy\n'
        'relative to a standing calibration, hysteresis\n'
        'gate threshold held-out; transition test\n'
        '(standing -> walking junction)', fs=FS)
    arrow(ax, 2.55, 7.2, 2.55, 6.9); arrow(ax, 7.45, 7.2, 7.45, 6.9)
    box(ax, 0.2, 3.65, 9.6, 1.1, 'Fixed decoders of the dataset authors\n'
        'ERP: shrinkage LDA trained on the standing session, tested at each speed -> AUC\n'
        'SSVEP: training-free CCA on 5-s epochs, three classes -> accuracy', fc='#eef7ee', ec='#275', fs=FS)
    arrow(ax, 2.55, 5.1, 2.55, 4.75); arrow(ax, 7.45, 5.1, 7.45, 4.75)
    box(ax, 0.2, 1.65, 3.05, 1.65, 'Parameters\ntwo participant halves\n(fixed seed)\nmu, r, theta chosen on one\nhalf, applied to the other,\nthen swapped', fc='#fff6e6', ec='#a60', fs=FS)
    box(ax, 3.45, 1.65, 3.1, 1.65, 'Inference\nfamilies F1-F14 fixed before\nthe held-out runs\nWilcoxon + Holm; bootstrap CI;\nHodges-Lehmann; up/down\ncounts', fc='#fff6e6', ec='#a60', fs=FS)
    box(ax, 6.75, 1.65, 3.05, 1.65, 'Controls\nstanding cost of every method\nsurrogate references\n(time shift, phase-randomized)\nIMU-only decoding\nsingle-release check (S14)', fc='#fff6e6', ec='#a60', fs=FS)
    arrow(ax, 1.7, 3.65, 1.7, 3.3); arrow(ax, 5.0, 3.65, 5.0, 3.3); arrow(ax, 8.3, 3.65, 8.3, 3.3)
    box(ax, 0.2, 0.3, 9.6, 0.95, 'Outputs regenerated from per-participant results by one script:\nall tables (main text and S1-S17), figures and the public repository', fc='#eeeeee', ec='#555', fs=FS)
    arrow(ax, 5.0, 1.65, 5.0, 1.25)
    save(fig, 'fig_protocol')
    print('fig_protocol done', flush=True)

if 'forest' in FIGS:
    # ---------------- fig_forest ----------------
    COND = [('ERP', '0.8'), ('ERP', '1.6'), ('ERP', '2.0'), ('SSVEP', '0.8'), ('SSVEP', '1.6'), ('SSVEP', '2.0')]
    rows = [('F1 reg vs none', 'reg', 'reg − none', 'tab:blue'),
            ('F2 gated NLMS vs none', 'nlms_ho', 'gated NLMS − none', 'tab:red'),
            ('F11 blockwise reg vs reg', 'reg_b30', 'reg 30 s − reg', 'tab:purple'),
            ('F4 other offline vs none', 'asr10_std', 'asr10 − none', 'tab:gray'),
            ('F4 other offline vs none', 'cca_ho', 'cca − none', 'tab:green'),
            ('F4 other offline vs none', 'gait+reg', 'gait+reg − none', 'tab:orange')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    ylabels = []
    yy = 0
    for ax_i, task in enumerate(['ERP', 'SSVEP']):
        ax = axes[ax_i]
        yy = 0
        for fam, meth, lab, col in rows:
            for sp in ['0.8', '1.6', '2.0']:
                r = next((x for x in S[fam] if x['b'] == meth and x['task'] == task and x['speed'] == sp), None)
                if r is None:
                    continue
                filled = r['p_holm'] < 0.05
                ax.errorbar(r['mean'], yy, xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]], fmt='o', color=col, mfc=col if filled else 'white', ms=5, capsize=2, lw=1)
                if ax_i == 0:
                    ylabels.append(f'{lab}, {sp} m/s')
                yy += 1
            yy += 0.6
        # 站立代价(F5)
        for meth, lab, col in [('nlms', 'NLMS − none, standing', 'tab:red'), ('reg_b30', 'reg 30 s − none, standing', 'tab:purple'), ('reg', 'reg − none, standing', 'tab:blue'), ('asr10_std', 'asr10 − none, standing', 'tab:gray')]:
            r = next((x for x in S['F5 standing cost'] if x['b'] == meth and x['task'] == task), None)
            if r is None:
                continue
            filled = r['p_holm'] < 0.05
            ax.errorbar(r['mean'], yy, xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]], fmt='s', color=col, mfc=col if filled else 'white', ms=5, capsize=2, lw=1)
            if ax_i == 0:
                ylabels.append(lab)
            yy += 1
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('ERP change (AUC × 100 points)' if task == 'ERP' else 'SSVEP change (percentage points)')
        ax.grid(axis='x', alpha=.3)
        ax.set_title(task, fontsize=9)
    # y 轴刻度:重算位置
    pos = []
    yy = 0
    for fam, meth, lab, col in rows:
        for sp in ['0.8', '1.6', '2.0']:
            pos.append(yy); yy += 1
        yy += 0.6
    for _ in range(4):
        pos.append(yy); yy += 1
    axes[0].set_yticks(pos); axes[0].set_yticklabels(ylabels, fontsize=7)
    axes[0].invert_yaxis()
    fig.text(0.5, 0.005, 'Mean paired difference with participant-bootstrap 95% CI; filled markers: Holm-significant within family', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, 'fig_forest')
    print('fig_forest done', flush=True)


import p1_clean_eval as p  # noqa: E402
import p1_causal_replay as c  # noqa: E402
if 'gate' in FIGS:
    # ---------------- fig_gate (recompute one participant) ----------------

    SUB = 's03'
    g0 = c.causal_eeg((SUB, 'SSVEP', '0.0')); g1 = c.causal_eeg((SUB, 'SSVEP', '1.6')); gt = c.causal_eeg((SUB, 'ERP', 'tr'))
    X0, clab, on0, y0, R0 = g0; X1, _, on1, y1, R1 = g1
    calib_energy = float(np.median(c.motion_energy(gt[4])[5 * c.FS:]))
    X = np.concatenate([X0, X1], 1); Rraw = np.concatenate([R0, R1], 1)
    e = c.motion_energy(Rraw) / max(calib_energy, 1e-30)
    gate = c.trailing_gate(Rraw, calib_energy, thr_on=100.0, thr_off=40.0)
    Xc = c.nlms_causal(X, c.running_norm(Rraw), gate, 0.05)
    T0 = X0.shape[1]; t = np.arange(X.shape[1]) / c.FS; tj = T0 / c.FS
    first_on = tj + (np.argmax(gate[T0:] > 0) / c.FS if (gate[T0:] > 0).any() else np.nan)
    oz = clab.index('Oz')
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=False)
    ax = axes[0]
    ax.semilogy(t, np.maximum(e, 1e-3), color='tab:green', lw=0.8)
    ax.axhline(100, color='k', ls='--', lw=0.8, label='θ = 100 (on)'); ax.axhline(40, color='gray', ls=':', lw=0.8, label='0.4θ (off)')
    ax.axvline(tj, color='tab:red', lw=0.8); ax.set_ylabel('Motion energy ratio\n(head, 0.7–4 Hz, 2-s trailing)')
    ax.legend(fontsize=7, loc='upper left', frameon=False); ax.set_title(f'Participant {SUB}: standing SSVEP recording followed by the 1.6 m/s recording (junction at {tj:.0f} s)', fontsize=8)
    ax = axes[1]
    ax.fill_between(t, 0, gate, step='post', color='tab:red', alpha=.5); ax.set_ylim(-0.1, 1.2); ax.set_yticks([0, 1]); ax.set_yticklabels(['off', 'on'])
    ax.axvline(tj, color='tab:red', lw=0.8); ax.set_ylabel('Gate state'); ax.set_xlabel('Time (s)')
    ax.annotate(f'opens {first_on - tj:.1f} s after the junction', xy=(first_on, 1.0), xytext=(first_on + 40, 1.08), fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.7))
    for a in axes[:2]:
        a.set_xlim(0, t[-1])
    ax = axes[2]
    w0, w1 = int((tj - 6) * c.FS), int((tj + 14) * c.FS)
    ax.plot(t[w0:w1], X[oz, w0:w1], color='k', lw=0.7, label='Oz, causal front end')
    ax.plot(t[w0:w1], Xc[oz, w0:w1], color='tab:red', lw=0.7, alpha=.8, label='Oz, gated NLMS')
    ax.axvline(tj, color='tab:red', lw=0.8); ax.axvline(first_on, color='tab:red', ls=':', lw=0.8)
    ax.set_xlim(t[w0], t[w1 - 1]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Oz (µV)'); ax.legend(fontsize=7, loc='upper left', frameon=False)
    ax.text(tj + 0.3, ax.get_ylim()[1] * 0.85, 'walking starts', fontsize=7, color='tab:red'); ax.text(first_on + 0.3, ax.get_ylim()[1] * 0.85, 'gate opens', fontsize=7, color='tab:red')
    fig.tight_layout(); save(fig, 'fig_gate')
    print('fig_gate done', flush=True)


if 'probe' in FIGS:
    # ---------------- fig_probe_topo ----------------
    from joblib import Parallel, delayed  # noqa: E402
    import mne  # noqa: E402
    import glob  # noqa: E402
    mne.set_log_level('ERROR')
    P = json.load(open(os.path.join(ROOT, 'results', 'p1_clean_eval', 'ssvep_signal_probe.json')))['summary']
    subs = [f's{i:02d}' for i in range(1, 19)] + sorted(os.path.basename(d) for d in glob.glob(os.path.join(p.OSF, 'sub-*')) if os.path.isdir(d))


    def removed_var(sub):
        out = {}
        for sp in ['0.8', '1.6', '2.0']:
            g = p.get_data(sub, 'ERP', sp)
            if g is None:
                continue
            X, clab, on, y, R = g
            Xr = p.m_reg(X, R)
            out[sp] = (clab, 1 - Xr.var(1) / X.var(1))
        return sub, out


    res = dict(Parallel(n_jobs=N_JOBS)(delayed(removed_var)(s) for s in subs))
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    sp_ = ['0.0', '0.8', '1.6', '2.0']
    for key, lab, col, ls in [('none_corr_true', 'attended freq., no processing', 'k', '-'), ('none_corr_wrong', 'best other freq., no processing', 'k', '--'),
                              ('reg_corr_true', 'attended, reg', 'tab:blue', '-'), ('reg_corr_wrong', 'best other, reg', 'tab:blue', '--'),
                              ('nlms_gated_corr_true', 'attended, gated NLMS', 'tab:red', '-'), ('nlms_gated_corr_wrong', 'best other, gated NLMS', 'tab:red', '--')]:
        v = [P[f'{s}_{key}']['mean'] for s in sp_]
        ax.plot(range(4), v, color=col, ls=ls, marker='o', ms=3, lw=1, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(['0', '0.8', '1.6', '2.0']); ax.set_xlabel('Treadmill speed (m/s)'); ax.set_ylabel('Mean CCA correlation per trial')
    ax.legend(fontsize=5.5, frameon=False, loc='center right'); ax.grid(alpha=.3); ax.set_title('SSVEP signal probe', fontsize=8)
    vmax = 0
    maps = {}
    for sp in ['0.8', '1.6', '2.0']:
        arrs = [v[sp][1] for v in res.values() if sp in v]
        clab = next(v[sp][0] for v in res.values() if sp in v)
        m = np.mean(arrs, 0); maps[sp] = (clab, m, len(arrs)); vmax = max(vmax, m.max())
    info = mne.create_info(maps['1.6'][0], 100, 'eeg'); info.set_montage('standard_1020', on_missing='ignore')
    for k, sp in enumerate(['0.8', '1.6', '2.0']):
        ax = fig.add_subplot(gs[0, k + 1])
        clab, m, n = maps[sp]
        im, _ = mne.viz.plot_topomap(m, info, axes=ax, show=False, cmap='Reds', vlim=(0, vmax), contours=4, sensors=True)
        ax.set_title(f'{sp} m/s (n={n})', fontsize=8)
    cb = fig.colorbar(im, ax=fig.axes[1:], shrink=0.7, pad=0.02); cb.set_label('Fraction of EEG variance removed by reg', fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.text(0.62, 0.95, 'Variance removed by lagged IMU regression (ERP recordings)', ha='center', fontsize=8)
    save(fig, 'fig_probe_topo')
    json.dump({sp: {'channels': maps[sp][0], 'mean_removed_fraction': maps[sp][1].tolist(), 'n': maps[sp][2]} for sp in maps}, open(os.path.join(ROOT, 'results', 'p1_clean_eval', 'removed_variance_topo.json'), 'w'), indent=1)
    print('fig_probe_topo done', flush=True)

