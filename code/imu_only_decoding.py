#!/usr/bin/env python3
"""对照:只用惯性传感器信号能不能解码任务?若能,说明任务与运动相关,惯性参考回归可能"顺带"注入任务信息。
ERP:用惯性信号(18 路)在 -200~800 ms 内每 50 ms 的均值作特征,站立训练段训 LDA,各速度测 AUC(与脑电同一套流程)。
     站立训练段里惯性几乎不动,所以也报在各速度段内 5 折交叉验证的 AUC。
SSVEP:对惯性信号做与脑电相同的 CCA 三分类,报准确率(期望 ≈33%)。"""
import glob, json, os, re, sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_clean_eval as p

def erp_feats_imu(R, on, y):
    t = np.arange(-20, 80) * 10
    ep = np.stack([R[:, o - 20:o + 80] for o in on if o - 20 >= 0 and o + 80 <= R.shape[1]], -1)
    yy = y[0][:ep.shape[-1]]
    ep = ep - ep[:, (t >= -200) & (t <= 0), :].mean(1, keepdims=True)
    f = np.stack([ep[:, (t > a) & (t <= b), :].mean(1) for a, b in p.ERP_IVALS], 0)
    return f.reshape(-1, f.shape[-1]).T, yy

subs = sorted({re.match(r'(s\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(p.DATA, 's*_scalp_*.mat'))}) + [f'sub-{i}' for i in range(19, 25)]
res = {}
for sub in subs:
    r = {}
    g = p.get_data(sub, 'ERP', 'tr')
    clf = None
    if g is not None:
        X, clab, on, y, R = g
        f, yy = erp_feats_imu(R, on, y)
        clf = LDA(solver='lsqr', shrinkage='auto').fit(f, yy)
    for sp in p.SPEEDS:
        g = p.get_data(sub, 'ERP', sp)
        if g is not None:
            X, clab, on, y, R = g
            f, yy = erp_feats_imu(R, on, y)
            if clf is not None:
                r[f'ERP_{sp}_imu_tr'] = float(roc_auc_score(yy, clf.decision_function(f)))
            pred = np.zeros(len(yy))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(f, yy):
                pred[te] = LDA(solver='lsqr', shrinkage='auto').fit(f[tr], yy[tr]).decision_function(f[te])
            r[f'ERP_{sp}_imu_cv'] = float(roc_auc_score(yy, pred))
        g = p.get_data(sub, 'SSVEP', sp)
        if g is not None:
            X, clab, on, y, R = g
            n = 5 * p.FS; tt = np.arange(1, n + 1) / p.FS
            refs = [np.stack([np.sin(2*np.pi*60/k*tt), np.cos(2*np.pi*60/k*tt), np.sin(4*np.pi*60/k*tt), np.cos(4*np.pi*60/k*tt)], 1) for k in p.SSVEP_DIV]
            pred, lab = [], []
            for i, o in enumerate(on):
                if o + n > R.shape[1]: continue
                pred.append(int(np.argmax([p.cca_mean_corr(R[:, o:o + n].T, rr) for rr in refs]))); lab.append(int(np.argmax(y[:, i])))
            r[f'SSVEP_{sp}_imu'] = float(np.mean(np.array(pred) == np.array(lab)) * 100)
    res[sub] = r
    print(sub, {k: round(v, 3) for k, v in r.items()}, flush=True)
summ = {}
for key in sorted({k for r in res.values() for k in r}):
    v = [r[key] for r in res.values() if key in r]
    summ[key] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)}
    print(f'{key:18s} {np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} (n={len(v)})')
os.makedirs(p.OUT, exist_ok=True)
json.dump({'per_subject': res, 'summary': summ}, open(os.path.join(p.OUT, 'imu_only_decoding.json'), 'w'), indent=1)
