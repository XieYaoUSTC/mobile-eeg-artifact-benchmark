#!/usr/bin/env python3
"""审稿补算:走路速度 SSVEP 损失 7–8 点无方法可救——是"信号弱了"还是"分类器受扰"?
对每人每速度,算每个 5 s 试次与正确刺激频率参考的 CCA 相关(信号强度代理)与最强错误参考的相关(干扰代理),
取均值;再给 margin = 正确 − 最强错误。不处理 / 回归 / 门控 NLMS 三列。
输出 results/p1_clean_eval/ssvep_signal_probe.json
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_clean_eval as p  # noqa: E402
from baseline_mobile_bci import SSVEP_CH, SSVEP_DIV, cca_mean_corr  # noqa: E402

FS = p.FS


def probe(X, clab, on, y):
    idx = [clab.index(c) for c in SSVEP_CH]
    n = 5 * FS
    tt = np.arange(1, n + 1) / FS
    refs = [np.stack([np.sin(2 * np.pi * 60 / k * tt), np.cos(2 * np.pi * 60 / k * tt),
                      np.sin(4 * np.pi * 60 / k * tt), np.cos(4 * np.pi * 60 / k * tt)], 1) for k in SSVEP_DIV]
    corr_true, corr_wrong = [], []
    for i, o in enumerate(on):
        if o + n > X.shape[1]:
            continue
        c = [cca_mean_corr(X[idx, o:o + n].T, r) for r in refs]
        lab = int(np.argmax(y[:, i]))
        corr_true.append(c[lab])
        corr_wrong.append(max(c[j] for j in range(3) if j != lab))
    return float(np.mean(corr_true)), float(np.mean(corr_wrong)), float(np.mean(np.array(corr_true) - np.array(corr_wrong)))


def run_subject(sub):
    res = {}
    for sp in p.SPEEDS:
        g = p.get_data(sub, 'SSVEP', sp)
        if g is None:
            continue
        X, clab, on, y, R = g
        for m in ('none', 'reg', 'nlms_gated'):
            Xc = p.METHODS[m](X, R, None)
            ct, cw, mg = probe(Xc, clab, on, y)
            res[f'{sp}_{m}'] = {'corr_true': ct, 'corr_wrong': cw, 'margin': mg}
    print(sub, json.dumps({k: round(v['margin'], 3) for k, v in res.items()}), flush=True)
    return sub, res


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    subs = sorted({re.match(r'(s\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(p.DATA, 's*_scalp_*.mat'))})
    subs += sorted(os.path.basename(d) for d in glob.glob(os.path.join(p.OSF, 'sub-*')) if os.path.isdir(d))
    if os.environ.get('SUBS'):
        subs = [x for x in subs if x in os.environ['SUBS'].split(',')]
    rows = dict(Parallel(n_jobs=n_jobs)(delayed(run_subject)(s) for s in subs))
    summ = {}
    for sp in p.SPEEDS:
        for m in ('none', 'reg', 'nlms_gated'):
            for k in ('corr_true', 'corr_wrong', 'margin'):
                v = [r[f'{sp}_{m}'][k] for r in rows.values() if f'{sp}_{m}' in r]
                if v:
                    summ[f'{sp}_{m}_{k}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)}
    # 配对:各速度 vs 站立(不处理)
    paired = {}
    for sp in ('0.8', '1.6', '2.0'):
        for k in ('corr_true', 'corr_wrong', 'margin'):
            ks = [s for s in rows if f'0.0_none' in rows[s] and f'{sp}_none' in rows[s]]
            d = np.array([rows[s][f'{sp}_none'][k] - rows[s]['0.0_none'][k] for s in ks])
            paired[f'{sp}_vs_standing_{k}'] = {'n': len(d), 'mean': float(d.mean()), 'up': int((d > 0).sum()), 'down': int((d < 0).sum())}
    json.dump({'per_subject': rows, 'summary': summ, 'paired_vs_standing_none': paired}, open(os.path.join(p.OUT, 'ssvep_signal_probe.json'), 'w'), indent=1)
    print('\n=== SSVEP 信号探针(不处理):速度  正确参考相关  最强错误相关  margin ===')
    for sp in p.SPEEDS:
        s = summ
        print(f"  {sp} m/s  {s[f'{sp}_none_corr_true']['mean']:.3f}  {s[f'{sp}_none_corr_wrong']['mean']:.3f}  {s[f'{sp}_none_margin']['mean']:.3f}  (n={s[f'{sp}_none_margin']['n']})")
    print('配对 vs 站立:', {k: round(v['mean'], 3) for k, v in paired.items()})


if __name__ == '__main__':
    main()
