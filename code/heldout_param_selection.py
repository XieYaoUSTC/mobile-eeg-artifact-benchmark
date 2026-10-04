#!/usr/bin/env python3
"""留人参数选择:把 24 人按固定种子分成两半,在 A 半上选参数、在 B 半上评,再交换;报合并后的留出结果。
选择目标(预先定下):各行走/慢跑条件(ERP 0.8/1.6/2.0 与 SSVEP 0.8/1.6/2.0)上「方法 − 不处理」的平均提升,
ERP 的 AUC 乘 100 换成"点",与 SSVEP 的百分点等权平均。站立段不进目标(门控后不处理)。
参数:NLMS 步长 ∈ {0.02, 0.05, 0.10, 0.20};cca 阈值 ∈ {0.2, 0.3, 0.4}。数据来自已有扫描结果文件。
另给:全体 24 人上直接选(即事后选择)的参数与留出结果的差,说明事后选择的乐观量。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'p1_clean_eval')
base = json.load(open(os.path.join(R, 'results_24.json')))['per_subject']
tun = json.load(open(os.path.join(R, 'results_24_tuning.json')))['per_subject']
data = {s: {**base.get(s, {}), **tun.get(s, {})} for s in set(base) | set(tun)}
subs = sorted(data)
rng = np.random.default_rng(20261004)
perm = rng.permutation(len(subs))
halves = [sorted(subs[i] for i in perm[:12]), sorted(subs[i] for i in perm[12:])]
COND = [('ERP', '0.8'), ('ERP', '1.6'), ('ERP', '2.0'), ('SSVEP', '0.8'), ('SSVEP', '1.6'), ('SSVEP', '2.0')]
GRIDS = {'nlms': {'0.02': 'nlms_mu02', '0.05': 'nlms_gated', '0.10': 'nlms_mu10', '0.20': 'nlms_mu20'},
         'cca': {'0.2': 'cca20', '0.3': 'cca', '0.4': 'cca40'}}


def gain(sub, task, sp, key):
    d = data.get(sub, {})
    k0, k1 = f'{task}_{sp}_none', f'{task}_{sp}_{key}'
    if k0 not in d or k1 not in d:
        return None
    g = d[k1] - d[k0]
    return g * 100 if task == 'ERP' else g


def objective(subset, key):
    vals = []
    for task, sp in COND:
        g = [gain(s, task, sp, key) for s in subset]
        g = [x for x in g if x is not None]
        if g:
            vals.append(np.mean(g))
    return float(np.mean(vals))


out = {'halves': halves, 'conditions': COND}
for fam, grid in GRIDS.items():
    rows = {}
    heldout = {}
    for i, (sel, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        scores = {pv: objective(sel, key) for pv, key in grid.items()}
        best = max(scores, key=scores.get)
        rows[f'fold{i + 1}'] = {'select_on': 'A' if i == 0 else 'B', 'scores': scores, 'chosen': best}
        for s in test:
            heldout[s] = grid[best]
    # 留出结果:每人用"在另一半上选出的参数"
    res = {}
    for task, sp in COND + [('ERP', '0.0'), ('SSVEP', '0.0')]:
        g = [gain(s, task, sp, heldout[s]) for s in subs]
        g = [x for x in g if x is not None]
        res[f'{task}_{sp}'] = {'heldout_mean_gain': float(np.mean(g)), 'n': len(g)}
    insample = {pv: objective(subs, key) for pv, key in grid.items()}
    best_all = max(insample, key=insample.get)
    rows['insample_all24'] = {'scores': insample, 'chosen': best_all}
    rows['heldout_gain_by_condition'] = res
    rows['heldout_objective'] = float(np.mean([res[f'{t}_{s}']['heldout_mean_gain'] for t, s in COND]))
    rows['insample_objective'] = insample[best_all]
    out[fam] = rows
    print(f'== {fam}: 两折各选 {rows["fold1"]["chosen"]} / {rows["fold2"]["chosen"]};全体事后选 {best_all}')
    print(f'   目标值: 留出 {rows["heldout_objective"]:.2f} 点  vs  事后(乐观) {rows["insample_objective"]:.2f} 点')
    for t, s in COND:
        print(f'   {t} {s}: 留出平均提升 {res[f"{t}_{s}"]["heldout_mean_gain"]:+.2f} 点 (n={res[f"{t}_{s}"]["n"]})')
json.dump(out, open(os.path.join(R, 'heldout_param_selection.json'), 'w'), indent=1)
