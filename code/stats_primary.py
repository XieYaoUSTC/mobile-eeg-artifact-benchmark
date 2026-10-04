#!/usr/bin/env python3
"""主对照的统计:逐人配对 Wilcoxon(精确法,零差值按 'zsplit' 处理)+ 家族内 Holm 校正 + 逐人 bootstrap 95% CI。
家族(预先声明):
  F1 离线惯性回归 reg vs none(6 个行走/慢跑条件)
  F2 门控在线 NLMS vs none(6 条件)
  F3 门控在线 NLMS vs reg(6 条件)
  F4 其余离线方法 vs none(cca、gait、gait+reg、asr × 6 条件 = 24 个)
  F5 站立段:nlms(不门控)与 nlms_gated vs none(ERP/SSVEP 各 1)
输出 results/p1_clean_eval/stats_primary.md 与 .json
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'p1_clean_eval')
base = json.load(open(os.path.join(R, 'results_24.json')))['per_subject']
gated = json.load(open(os.path.join(R, 'results_24_gated.json')))['per_subject']
asrfix = json.load(open(os.path.join(R, 'results_24_asrfix_icc.json')))['per_subject']      # 修复秩亏后的 ASR 与 iCanClean 改编版,覆盖旧 asr 列
data = {s: {**base.get(s, {}), **gated.get(s, {}), **asrfix.get(s, {})} for s in set(base) | set(gated) | set(asrfix)}
COND = [('ERP', '0.8'), ('ERP', '1.6'), ('ERP', '2.0'), ('SSVEP', '0.8'), ('SSVEP', '1.6'), ('SSVEP', '2.0')]
rng = np.random.default_rng(1)


def paired(task, sp, a, b):
    xa, xb = [], []
    for d in data.values():
        ka, kb = f'{task}_{sp}_{a}', f'{task}_{sp}_{b}'
        if ka in d and kb in d:
            xa.append(d[ka]); xb.append(d[kb])
    xa, xb = np.array(xa), np.array(xb)
    diff = xb - xa
    if task == 'ERP':
        diff = diff * 100
    n = len(diff)
    if n == 0:
        return None
    p = wilcoxon(diff, zero_method='zsplit', method='exact' if n <= 25 else 'approx').pvalue if np.any(diff != 0) else 1.0
    boot = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(10000)])
    return {'n': n, 'mean': float(diff.mean()), 'ci_lo': float(np.percentile(boot, 2.5)), 'ci_hi': float(np.percentile(boot, 97.5)),
            'n_up': int((diff > 0).sum()), 'n_down': int((diff < 0).sum()), 'p': float(p)}


def holm(ps):
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


families = {
    'F1 reg vs none': [(t, s, 'none', 'reg') for t, s in COND],
    'F2 nlms_gated vs none': [(t, s, 'none', 'nlms_gated') for t, s in COND],
    'F3 nlms_gated vs reg': [(t, s, 'reg', 'nlms_gated') for t, s in COND],
    'F4 other offline vs none': [(t, s, 'none', m) for m in ('cca', 'gait', 'gait+reg', 'asr', 'asr10', 'icc_w4') for t, s in COND],
    'F5 standing': [('ERP', '0.0', 'none', 'nlms'), ('SSVEP', '0.0', 'none', 'nlms'), ('ERP', '0.0', 'none', 'nlms_gated'), ('SSVEP', '0.0', 'none', 'nlms_gated')],
}
out = {}
lines = ['# 主对照统计(逐人配对;ERP 的差以 AUC×100 计,SSVEP 以百分点计;Holm 校正在家族内)', '']
for fam, tests in families.items():
    res = []
    for t, s, a, b in tests:
        r = paired(t, s, a, b)
        if r:
            r.update({'task': t, 'speed': s, 'a': a, 'b': b})
            res.append(r)
    adj = holm(np.array([r['p'] for r in res]))
    for r, pa in zip(res, adj):
        r['p_holm'] = float(pa)
    out[fam] = res
    lines += [f'## {fam}', '', '| 条件 | 对比 | n | 平均差 | 95% CI | 变好/变差 | p | p(Holm) |', '|---|---|---|---|---|---|---|---|']
    for r in res:
        lines.append(f"| {r['task']} {r['speed']} | {r['b']} − {r['a']} | {r['n']} | {r['mean']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['n_up']}/{r['n_down']} | {r['p']:.3g} | {r['p_holm']:.3g} |")
    lines.append('')
json.dump(out, open(os.path.join(R, 'stats_primary.json'), 'w'), indent=1, ensure_ascii=False)
open(os.path.join(R, 'stats_primary.md'), 'w').write('\n'.join(lines))
print('\n'.join(lines))
