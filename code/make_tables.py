#!/usr/bin/env python3
"""论文一 · 从结果文件一次性生成全部表格(单一数据源),输出 docs/tables_v2.md 与 results/tables_v2.json。
规则:所有差值、CI、p 值都用未取整的逐人数据计算;显示时四舍五入到小数后两位(差值)或三位(均值)。
一致性:sub-21 的站立 ERP 段缺惯性通道,在所有表中一律排除(与因果回放、门控版一致)。
"""
import glob
import json
import os
import re

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'p1_clean_eval')
RC = os.path.join(ROOT, 'results', 'p1_causal_replay')
rng = np.random.default_rng(7)
SPEEDS = ['0.0', '0.8', '1.6', '2.0']
COND = [('ERP', '0.8'), ('ERP', '1.6'), ('ERP', '2.0'), ('SSVEP', '0.8'), ('SSVEP', '1.6'), ('SSVEP', '2.0')]
FILES = ['results_24.json', 'results_24_gated.json', 'results_24_asrfix_icc.json', 'results_24_tuning.json',
         'results_24_ablation.json', 'results_24_controls.json', 'results_24_controls2.json']

data = {}
for f in FILES:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        continue
    for s, d in json.load(open(p))['per_subject'].items():
        data.setdefault(s, {}).update(d)
for s in data:                                       # 一致性排除
    if s == 'sub-21':
        data[s] = {k: v for k, v in data[s].items() if not k.startswith('ERP_0.0_')}
subs = sorted(data)


def col(task, sp, m):
    return {s: data[s][f'{task}_{sp}_{m}'] for s in subs if f'{task}_{sp}_{m}' in data[s]}


def mean_str(task, sp, m):
    v = list(col(task, sp, m).values())
    if not v:
        return '—'
    return f'{np.mean(v):.3f}' if task == 'ERP' else f'{np.mean(v):.1f}'


def paired(task, sp, a, b):
    ca, cb = col(task, sp, a), col(task, sp, b)
    ks = sorted(set(ca) & set(cb))
    d = np.array([cb[k] - ca[k] for k in ks])
    if task == 'ERP':
        d = d * 100
    n = len(d)
    if n == 0:
        return None
    nz = d[d != 0]
    p = float(wilcoxon(d, zero_method='zsplit', method='exact').pvalue) if len(nz) else 1.0
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
    return {'n': n, 'mean': float(d.mean()), 'lo': float(np.percentile(boot, 2.5)), 'hi': float(np.percentile(boot, 97.5)),
            'up': int((d > 0).sum()), 'down': int((d < 0).sum()), 'tie': int((d == 0).sum()), 'p': p}


def holm(ps):
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def fmt(r):
    if r is None:
        return '—'
    return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]; {r['up']}/{r['down']}" + (f"/{r['tie']}" if r['tie'] else '') + f"; p={r['p']:.2g}; Holm={r['p_holm']:.2g}"


out = {}
L = ['# 论文一 表格(由 make_tables.py 从结果文件生成;差值以 ERP AUC×100 点 / SSVEP 百分点计)', '']

# ---- S2 人数表
L += ['## 表 S2 各条件纳入人数(不处理列有值即纳入)', '', '| 条件 | ERP n | SSVEP n | 缺的受试者(ERP / SSVEP) |', '|---|---|---|---|']
incl = {}
for sp in SPEEDS:
    e, v = col('ERP', sp, 'none'), col('SSVEP', sp, 'none')
    me = sorted(set(subs) - set(e)); mv = sorted(set(subs) - set(v))
    incl[sp] = {'ERP': len(e), 'SSVEP': len(v), 'missing_ERP': me, 'missing_SSVEP': mv}
    L.append(f"| {sp} m/s | {len(e)} | {len(v)} | {', '.join(me) or '—'} / {', '.join(mv) or '—'} |")
out['inclusion'] = incl
L.append('')

# ---- 表 1 均值
METHODS1 = [('none', 'none'), ('reg', 'reg'), ('asr', 'asr'), ('asr10', 'asr10'), ('cca40', 'cca r=0.4'), ('icc_w4', 'icc 4 s'), ('gait', 'gait'), ('gait+reg', 'gait+reg'), ('nlms_gated', 'gated NLMS (replay)')]
L += ['## 表 1 回顾性流水线各方法均值', '']
for task in ('ERP', 'SSVEP'):
    L += ['| ' + task + ' | ' + ' | '.join(lab for _, lab in METHODS1) + ' |', '|' + '---|' * (len(METHODS1) + 1)]
    for sp in SPEEDS:
        n = len(col(task, sp, 'none'))
        L.append(f'| {sp} m/s ({n}) | ' + ' | '.join(mean_str(task, sp, m) for m, _ in METHODS1) + ' |')
    L.append('')
out['table1'] = {f'{t}_{sp}_{m}': mean_str(t, sp, m) for t in ('ERP', 'SSVEP') for sp in SPEEDS for m, _ in METHODS1}

# ---- 表 2 家族统计
families = {
    'F1 reg vs none': [(t, s, 'none', 'reg') for t, s in COND],
    'F2 gated NLMS vs none': [(t, s, 'none', 'nlms_gated') for t, s in COND],
    'F3 gated NLMS vs reg': [(t, s, 'reg', 'nlms_gated') for t, s in COND],
    'F4 other offline vs none': [(t, s, 'none', m) for m in ('asr', 'asr10', 'cca40', 'icc_w4', 'gait', 'gait+reg') for t, s in COND],
    'F5 standing cost': [('ERP', '0.0', 'none', 'nlms'), ('SSVEP', '0.0', 'none', 'nlms'), ('ERP', '0.0', 'none', 'nlms_gated'), ('SSVEP', '0.0', 'none', 'nlms_gated')],
    'F6 controls vs true reference (reg)': [(t, s, 'reg', m) for m in ('reg_shift10', 'reg_shift', 'reg_shift120', 'reg_surr') for t, s in COND],
    'F7 controls vs true reference (NLMS)': [(t, s, 'nlms_gated', m) for m in ('nlms_shift10', 'nlms_shift', 'nlms_shift120', 'nlms_surr') for t, s in COND],
    'F8 ablation vs 18-ch reg': [(t, s, 'reg', m) for m in ('reg_head', 'reg_ankle') for t, s in COND],
}
stats = {}
for fam, tests in families.items():
    rows = []
    for t, s, a, b in tests:
        r = paired(t, s, a, b)
        if r:
            r.update({'task': t, 'speed': s, 'a': a, 'b': b})
            rows.append(r)
    if not rows:
        continue
    adj = holm(np.array([r['p'] for r in rows]))
    for r, pa in zip(rows, adj):
        r['p_holm'] = float(pa)
    stats[fam] = rows
    L += [f'## 表 2 · {fam}', '', '| 条件 | 对比 | 平均差 [95% CI]; 升/降(/平); p; Holm |', '|---|---|---|']
    for r in rows:
        L.append(f"| {r['task']} {r['speed']} (n={r['n']}) | {r['b']} − {r['a']} | {fmt(r)} |")
    L.append('')
out['stats'] = stats

# ---- 表 3 参数扫描(全部 8 个条件)
L += ['## 表 3 参数扫描(门控 NLMS 步长;cca 阈值;icc 窗长;岭系数)', '']
sweeps = [('NLMS μ', [('none', 'none'), ('nlms_mu02', '0.02'), ('nlms_gated', '0.05'), ('nlms_mu10', '0.10'), ('nlms_mu20', '0.20')]),
          ('cca r', [('none', 'none'), ('cca20', '0.2'), ('cca', '0.3'), ('cca40', '0.4')]),
          ('icc 窗', [('none', 'none'), ('icc', '2 s'), ('icc_w4', '4 s')]),
          ('reg 岭系数', [('none', 'none'), ('reg_a1', '0.01%'), ('reg', '0.1%'), ('reg_a3', '1%')])]
for name, grid in sweeps:
    for task in ('ERP', 'SSVEP'):
        L += [f'| {name} · {task} | ' + ' | '.join(lab for _, lab in grid) + ' |', '|' + '---|' * (len(grid) + 1)]
        for sp in SPEEDS:
            L.append(f'| {sp} m/s | ' + ' | '.join(mean_str(task, sp, m) for m, _ in grid) + ' |')
        L.append('')

# ---- 表 5 对照(均值)
L += ['## 表 5 对照:时移/替代参考与只用惯性解码(均值)', '']
ctrl = [('none', 'none'), ('reg', 'reg 真参考'), ('reg_shift10', 'reg 移 10 s'), ('reg_shift', 'reg 移 30 s'), ('reg_shift120', 'reg 移 120 s'), ('reg_surr', 'reg 相位随机'),
        ('nlms_gated', 'NLMS 真参考'), ('nlms_shift10', 'NLMS 移 10 s'), ('nlms_shift', 'NLMS 移 30 s'), ('nlms_shift120', 'NLMS 移 120 s'), ('nlms_surr', 'NLMS 相位随机')]
for task in ('ERP', 'SSVEP'):
    L += [f'| {task} | ' + ' | '.join(lab for _, lab in ctrl) + ' |', '|' + '---|' * (len(ctrl) + 1)]
    for sp in SPEEDS:
        L.append(f'| {sp} m/s | ' + ' | '.join(mean_str(task, sp, m) for m, _ in ctrl) + ' |')
    L.append('')
imu = os.path.join(R, 'imu_only_decoding.json')
if os.path.exists(imu):
    summ = json.load(open(imu))['summary']
    L += ['只用惯性信号解码(均值 ± SD, n):', '']
    for k, v in sorted(summ.items()):
        L.append(f"- {k}: {v['mean']:.3f} ± {v['sd']:.3f} (n={v['n']})")
    L.append('')

# ---- 表 6 因果回放
cz = sorted(glob.glob(os.path.join(RC, 'results_thr100_mu0.05.json')))
if cz:
    C = json.load(open(cz[-1]))
    cs = C['per_subject']
    L += ['## 表 6 因果回放(零阶保持、门控阈值 100 = 留出两折所选)', '', '| 条件 | none | NLMS | gated NLMS | gated − none | NLMS − none |', '|---|---|---|---|---|---|']
    crows = []
    for task in ('ERP', 'SSVEP'):
        for sp in SPEEDS:
            v0 = {s: cs[s][f'{task}_{sp}_none'] for s in cs if f'{task}_{sp}_none' in cs[s]}
            if not v0:
                continue
            vals = {}
            for m in ('nlms_c', 'nlms_gated_c'):
                vm = {s: cs[s][f'{task}_{sp}_{m}'] for s in cs if f'{task}_{sp}_{m}' in cs[s]}
                ks = sorted(set(v0) & set(vm))
                d = np.array([vm[k] - v0[k] for k in ks]) * (100 if task == 'ERP' else 1)
                p = float(wilcoxon(d, zero_method='zsplit', method='exact').pvalue) if np.any(d != 0) else 1.0
                boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
                vals[m] = {'n': len(d), 'mean': float(d.mean()), 'lo': float(np.percentile(boot, 2.5)), 'hi': float(np.percentile(boot, 97.5)), 'up': int((d > 0).sum()), 'down': int((d < 0).sum()), 'tie': int((d == 0).sum()), 'p': p, 'task': task, 'speed': sp}
            crows.append(vals)
            f = (lambda x: f'{x:.3f}') if task == 'ERP' else (lambda x: f'{x:.1f}')
            L.append(f"| {task} {sp} (n={len(v0)}) | {f(np.mean(list(v0.values())))} | {f(np.mean([cs[s][f'{task}_{sp}_nlms_c'] for s in cs if f'{task}_{sp}_nlms_c' in cs[s]]))} | {f(np.mean([cs[s][f'{task}_{sp}_nlms_gated_c'] for s in cs if f'{task}_{sp}_nlms_gated_c' in cs[s]]))} | __G{len(crows)-1}__ | __U{len(crows)-1}__ |")
    # Holm:门控 vs none 作为一个家族(8 条件),不门控 vs none 另一家族
    for key, tag in (('nlms_gated_c', 'G'), ('nlms_c', 'U')):
        ps = np.array([v[key]['p'] for v in crows])
        adj = holm(ps)
        for i, (v, pa) in enumerate(zip(crows, adj)):
            v[key]['p_holm'] = float(pa)
            L = [ln.replace(f'__{tag}{i}__', fmt(v[key])) for ln in L]
    g = C['gate']
    L += ['', f"门控逐秒决策(以段速度为真值):TP {g['per_second_decisions']['TP']:.0f}, FN {g['per_second_decisions']['FN']:.0f}, FP {g['per_second_decisions']['FP']:.0f}, TN {g['per_second_decisions']['TN']:.0f};灵敏度 {g['sensitivity']:.3f},特异度 {g['specificity']:.3f}", '']
    # 逐人门控表现
    per = {}
    for seg in C['gate_segments']:
        per.setdefault(seg['sub'], {'mov_on': [], 'stand_on': []})
        (per[seg['sub']]['mov_on'] if seg['moving'] else per[seg['sub']]['stand_on']).append(seg['gate_on_frac'])
    L += ['逐人门控:行走段门开比例均值 / 站立段门开比例均值', '', '| 受试者 | 行走段开门 | 站立段开门 |', '|---|---|---|']
    for s in sorted(per):
        mo = np.mean(per[s]['mov_on']) if per[s]['mov_on'] else float('nan')
        so = np.mean(per[s]['stand_on']) if per[s]['stand_on'] else float('nan')
        L.append(f'| {s} | {mo:.2f} | {so:.2f} |')
    ts = C.get('transition_summary', {})
    L += ['', '过渡实验(站立段+快走段相接):' + '; '.join(f"{k}={v['mean']:.3f}" for k, v in ts.items()), '']
    out['causal'] = {'rows': crows, 'gate': g, 'transition': ts}

# ---- 表 7 ASR 修复前后
old = json.load(open(os.path.join(R, 'results_24.json')))['per_subject']
new = json.load(open(os.path.join(R, 'results_24_asrfix_icc.json')))['per_subject']
L += ['## 表 7 ASR(阈值 20)秩亏修复前后,ERP 1.6 m/s 均值', '', '| 组 | n | 不处理 | 修复前 ASR | 修复后 ASR |', '|---|---|---|---|---|']
for grp, sel in (('figshare 18', [s for s in subs if s.startswith('s') and not s.startswith('sub-')]), ('OSF 6', [s for s in subs if s.startswith('sub-')])):
    ks = [s for s in sel if 'ERP_1.6_asr' in old.get(s, {}) and 'ERP_1.6_asr' in new.get(s, {})]
    L.append(f"| {grp} | {len(ks)} | {np.mean([old[s]['ERP_1.6_none'] for s in ks]):.3f} | {np.mean([old[s]['ERP_1.6_asr'] for s in ks]):.3f} | {np.mean([new[s]['ERP_1.6_asr'] for s in ks]):.3f} |")
L.append('')

# ---- 表 8 耗时
tp = os.path.join(R, 'timing.json')
if os.path.exists(tp):
    t = json.load(open(tp))
    L += ['## 表 8 单核处理每秒数据耗时(ms)', '', '| ' + ' | '.join(t['ms_per_s']) + ' |', '|' + '---|' * len(t['ms_per_s']), '| ' + ' | '.join(f'{v:.1f}' for v in t['ms_per_s'].values()) + ' |', '']

json.dump(out, open(os.path.join(ROOT, 'results', 'tables_v2.json'), 'w'), indent=1, ensure_ascii=False)
open(os.path.join(ROOT, 'docs', 'tables_v2.md'), 'w').write('\n'.join(L))
print('\n'.join(L))
