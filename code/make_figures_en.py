#!/usr/bin/env python3
"""论文一 · 英文图(PDF+PNG)供 JNE 排版:fig1 速度曲线、fig2 逐人散点、fig3 波形、fig4 频谱。输出 paper_jne/figs/。"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper_jne', 'figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
R = {}
for f in ['results_24.json', 'results_24_gated.json', 'results_24_asrfix_icc.json', 'results_24_tuning.json', 'results_24_controls2.json', 'results_24_methods2.json', 'results_24_asr_std.json']:
    for s, d in json.load(open(os.path.join(ROOT, 'results', 'p1_clean_eval', f)))['per_subject'].items():
        if f == 'results_24_asr_std.json':
            d = {k: v for k, v in d.items() if not k.endswith('_none')}
        R.setdefault(s, {}).update(d)
R['sub-21'] = {k: v for k, v in R.get('sub-21', {}).items() if not k.startswith('ERP_0.0_')}
C = json.load(open(os.path.join(ROOT, 'results', 'p1_causal_replay', 'results_thr100_mu0.05.json')))['per_subject']
H = json.load(open(os.path.join(ROOT, 'results', 'p1_clean_eval', 'heldout_param_selection.json')))
for fam, key in (('cca', 'cca_ho'), ('nlms', 'nlms_ho')):
    for sub, pv in H[fam]['heldout_assignment'].items():
        m = H[fam]['grid'][pv]
        for k in list(R.get(sub, {})):
            if k.endswith('_' + m):
                R[sub][k[:-len(m)] + key] = R[sub][k]
CCA_LAB = 'Global CCA (held-out r = ' + '/'.join(sorted(set([H['cca']['fold1']['chosen'], H['cca']['fold2']['chosen']]))) + ')'
speeds = ['0.0', '0.8', '1.6', '2.0']
labels = ['Standing\n0', 'Slow walk\n0.8', 'Fast walk\n1.6', 'Slight run\n2.0']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200)
    plt.close(fig)


# ---- Fig 1
methods = [('none', 'No processing', 'k', '-', 2.0), ('reg', 'IMU regression (reg)', 'tab:blue', '-', 2.0), ('nlms_ho', 'Gated NLMS (replay; standing = none by construction)', 'tab:red', '-', 2.0),
           ('asr10_std', 'ASR (cutoff 10)', 'tab:gray', '-', 1.0), ('cca_ho', CCA_LAB, 'tab:green', '-', 1.0),
           ('gait', 'Gait template', 'tab:orange', '-', 1.0), ('icc_w4', 'iCanClean-style (4 s, no lags)', 'tab:purple', '--', 1.0)]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1))
for ax, task, ylab in [(axes[0], 'ERP', 'ERP decoding (AUC)'), (axes[1], 'SSVEP', 'SSVEP accuracy (%)')]:
    ns = []
    for m, lab, col, ls, lw in methods:
        mean, se, n = [], [], []
        for sp in speeds:
            v = [d[f'{task}_{sp}_{m}'] for d in R.values() if f'{task}_{sp}_{m}' in d]
            mean.append(np.mean(v)); se.append(np.std(v, ddof=1) / np.sqrt(len(v))); n.append(len(v))
        ns = n
        ax.errorbar(range(4), mean, yerr=se, label=lab, color=col, ls=ls, lw=lw, marker='o', ms=3.5 if lw > 1.5 else 2.5, capsize=2, alpha=1 if lw > 1.5 else 0.8)
    for key, col, lab in (('nlms_gated_c', 'tab:red', 'Gated NLMS (causal pipeline)'), ('none', 'k', 'No processing (causal front end)')):
        vv = [[d[f'{task}_{sp}_{key}'] for d in C.values() if f'{task}_{sp}_{key}' in d] for sp in speeds]
        ax.errorbar(range(4), [np.mean(v) for v in vv], yerr=[np.std(v, ddof=1) / np.sqrt(len(v)) for v in vv], color=col, ls=':', marker='^', ms=4, lw=1.1, capsize=2, label=lab)
    ax.text(-0.12, 1.02, '(a)' if task == 'ERP' else '(b)', transform=ax.transAxes, fontsize=10, weight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{l}\n(n={k})' for l, k in zip(labels, ns)], fontsize=7.5)
    ax.set_ylabel(ylab); ax.set_xlabel('Treadmill speed (m/s)'); ax.grid(alpha=.3)
    ax.axhline(33.3 if task == 'SSVEP' else 0.5, ls='--', c='gray', lw=.7)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7.5, loc='lower center', ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.2, 1, 1)); save(fig, 'fig1_speed_curves')

# ---- Fig 2
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
for k, (ax, (task, sp, m, lab)) in enumerate(zip(axes, [('ERP', '1.6', 'reg', 'Fast walk ERP: IMU regression'), ('ERP', '2.0', 'nlms_ho', 'Slight run ERP: gated NLMS'), ('SSVEP', '2.0', 'nlms_ho', 'Slight run SSVEP: gated NLMS')])):
    a, b = [], []
    for d in R.values():
        if f'{task}_{sp}_none' in d and f'{task}_{sp}_{m}' in d:
            a.append(d[f'{task}_{sp}_none']); b.append(d[f'{task}_{sp}_{m}'])
    a, b = np.array(a), np.array(b)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); pad = (hi - lo) * .08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--', c='gray', lw=.7)
    up = b > a
    ax.scatter(a[up], b[up], c='tab:red', marker='o', s=16, label='improved')
    ax.scatter(a[~up], b[~up], c='tab:blue', marker='v', s=18, label='not improved')
    mean_pts = np.mean(b - a) * (100 if task == 'ERP' else 1)
    ax.set_title(f'{lab}\n{up.sum()}/{len(a)} improved, mean {mean_pts:+.1f} points', fontsize=7.5)
    unit = 'ERP AUC' if task == 'ERP' else 'SSVEP acc. (%)'
    ax.set_xlabel(f'No processing ({unit})'); ax.set_ylabel(f'After processing ({unit})'); ax.grid(alpha=.3)
    ax.text(-0.3, 1.2, f'({chr(97 + k)})', transform=ax.transAxes, fontsize=10, weight='bold')
    if k == 0:
        ax.legend(fontsize=6.5, frameon=False, loc='lower right')
fig.tight_layout(w_pad=2.0); save(fig, 'fig2_per_subject')

if os.environ.get('ONLY_FIG1') == '1':
    sys.exit(0)
# ---- Fig 3 and 4 (recompute from raw data)
import p1_clean_eval as p  # noqa: E402
from scipy.signal import welch  # noqa: E402

import glob  # noqa: E402
subs = [f's{i:02d}' for i in range(1, 19)] + sorted(os.path.basename(d) for d in glob.glob(os.path.join(p.OSF, 'sub-*')) if os.path.isdir(d))
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey='row')
t = np.arange(-20, 80) * 10
for r, sp in enumerate(['1.6', '2.0']):
    curves = {m: {'t': [], 'n': []} for m in ['none', 'reg', 'nlms_gated']}
    for sub in subs:
        g = p.get_data(sub, 'ERP', sp)
        if g is None:
            continue
        X, clab, on, y, Rr = g
        for m in curves:
            Xc = p.METHODS[m](X, Rr, None); pz = clab.index('Pz')
            ep = np.stack([Xc[pz, o - 20:o + 80] for o in on if o - 20 >= 0 and o + 80 <= Xc.shape[1]]); yy = y[0][:len(ep)]
            ep = ep - ep[:, (t >= -200) & (t <= 0)].mean(1, keepdims=True)
            curves[m]['t'].append(ep[yy == 1].mean(0)); curves[m]['n'].append(ep[yy == 0].mean(0))
    for k, (m, lab) in enumerate([('none', 'no processing'), ('reg', 'IMU regression'), ('nlms_gated', 'gated NLMS')]):
        ax = axes[r, k]; T = np.array(curves[m]['t']); N = np.array(curves[m]['n'])
        for arr, col, lb in ((T, 'tab:red', 'target'), (N, 'tab:blue', 'non-target')):
            mu, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
            ax.plot(t, mu, c=col, lw=1.2, label=lb); ax.fill_between(t, mu - se, mu + se, color=col, alpha=.2)
        ax.axvline(0, c='gray', lw=.7); ax.axvspan(200, 450, color='gray', alpha=.08); ax.grid(alpha=.3)
        ax.set_title(f'{"Fast walk 1.6 m/s" if sp == "1.6" else "Slight run 2.0 m/s"} (n={len(T)})\n{lab}', fontsize=7.5)
        ax.text(-0.18 if k == 0 else -0.08, 1.12, f'({chr(97 + 3 * r + k)})', transform=ax.transAxes, fontsize=10, weight='bold')
        if k == 0:
            ax.set_ylabel('Pz amplitude (µV)')
        if r == 1:
            ax.set_xlabel('Time after stimulus (ms)')
axes[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout(); save(fig, 'fig3_erp_waveforms')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for k, sp in enumerate(['0.8', '1.6', '2.0']):
    Pe, Pr, Pn, Pa = [], [], [], []
    for sub in subs:
        g = p.get_data(sub, 'SSVEP', sp)
        if g is None:
            continue
        X, clab, on, y, Rr = g; oz = clab.index('Oz')
        f, px = welch(X[oz], 100, nperseg=1000); Pe.append(px)
        f, pr = welch(p.METHODS['reg'](X, Rr)[oz], 100, nperseg=1000); Pr.append(pr)
        f, pn = welch(p.METHODS['nlms_gated'](X, Rr)[oz], 100, nperseg=1000); Pn.append(pn)
        f, pa = welch(np.sqrt((Rr[:3] ** 2).sum(0)), 100, nperseg=1000); Pa.append(pa)
    ax = axes[k]; m = (f >= 0.5) & (f <= 30)
    ax.semilogy(f[m], np.median(Pe, 0)[m], c='k', lw=1.1, label='Oz, no processing')
    ax.semilogy(f[m], np.median(Pr, 0)[m], c='tab:blue', lw=1.0, label='Oz, IMU regression')
    ax.semilogy(f[m], np.median(Pn, 0)[m], c='tab:red', lw=1.0, label='Oz, gated NLMS')
    ax2 = ax.twinx(); ax2.semilogy(f[m], np.median(Pa, 0)[m], c='tab:green', alpha=.6, lw=.9, label='head acceleration (right axis)'); ax2.tick_params(axis='y', labelsize=6, colors='tab:green')
    for fr in (5.45, 8.57, 12):
        ax.axvline(fr, c='gray', ls=':', lw=.7)
    ax.set_title(f'{sp} m/s (n={len(Pe)})', fontsize=8); ax.set_xlabel('Frequency (Hz)'); ax.grid(alpha=.3)
    ax.text(-0.25 if k == 0 else -0.12, 1.08, f'({chr(97 + k)})', transform=ax.transAxes, fontsize=10, weight='bold')
    if k == 0:
        ax.set_ylabel('Oz median PSD ($\\mu$V$^2$/Hz)'); ax.legend(fontsize=6, loc='upper right', frameon=False)
    if k == 2:
        ax2.set_ylabel('Head acceleration PSD (a.u.)', fontsize=7, color='tab:green')
        ax2.legend(fontsize=6, loc='upper right', frameon=False)
fig.tight_layout(); save(fig, 'fig4_spectra')
print('figures written to', OUT)
