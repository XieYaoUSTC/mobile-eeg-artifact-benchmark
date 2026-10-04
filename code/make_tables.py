#!/usr/bin/env python3
"""论文一 · 从结果文件一次性生成全部表格(单一数据源),输出 docs/tables_v3.md 与 results/tables_v3.json。
规则:所有差值、CI、p 值都用未取整的逐人数据计算;显示时四舍五入到小数后两位(差值)或三位(均值)。
一致性:sub-21 的站立 ERP 段缺惯性通道,在所有表中一律排除(与因果回放、门控版一致)。
v3(审稿后):
  - ASR 主口径改为标准顺序(ASR 在插值与 CAR 之前,asr_std/asr10_std);rank-safe 版降为敏感性(F14 直接对比)。
  - F5 站立代价扩到所有不门控的方法(nlms、reg、reg_b30、cca、asr_std、asr10_std、icc_w4);段级门控版因恒等不进家族。
  - 新家族:F11 分块回归 vs 整段回归;F12 门控 NLMS vs 30 s 分块回归;F13 CCA 2×2 vs none;F14 rank-safe vs 标准 ASR。
  - 主对比按 release 分层(figshare 18 人):F1f/F2f/F9f。
  - 每个配对比较同时给 Hodges–Lehmann 中位数位移及其 bootstrap 区间(与秩检验同一假设)。
  - CCA 阈值与 NLMS 步长按留人选参结果取"held-out"列(两折一致时等于单一参数)。
  - 因果回放前端敏感性(0.5 Hz vs 0.1 Hz 因果高通)。
"""
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'p1_clean_eval')
RC = os.path.join(ROOT, 'results', 'p1_causal_replay')
rng = np.random.default_rng(7)
SPEEDS = ['0.0', '0.8', '1.6', '2.0']
COND = [('ERP', '0.8'), ('ERP', '1.6'), ('ERP', '2.0'), ('SSVEP', '0.8'), ('SSVEP', '1.6'), ('SSVEP', '2.0')]
FILES = ['results_24.json', 'results_24_gated.json', 'results_24_asrfix_icc.json', 'results_24_tuning.json',
         'results_24_ablation.json', 'results_24_controls.json', 'results_24_controls2.json',
         'results_24_methods2.json', 'results_24_icc2.json', 'results_24_asr_std.json']

data = {}
for f in FILES:
    pth = os.path.join(R, f)
    if not os.path.exists(pth):
        print('missing', f)
        continue
    for s, d in json.load(open(pth))['per_subject'].items():
        if f == 'results_24_asr_std.json':
            d = {k: v for k, v in d.items() if not k.endswith('_none')}     # 其 none 列的插值口径不同,不覆盖主流水线
        data.setdefault(s, {}).update(d)
for s in data:                                       # 一致性排除
    if s == 'sub-21':
        data[s] = {k: v for k, v in data[s].items() if not k.startswith('ERP_0.0_')}
subs = sorted(data)
FIGSHARE = [s for s in subs if not s.startswith('sub-')]
OSF = [s for s in subs if s.startswith('sub-')]

# ---- 留人选参:生成 held-out 列(cca_ho / nlms_ho)
H = json.load(open(os.path.join(R, 'heldout_param_selection.json')))
HO = {}
for fam, key in (('cca', 'cca_ho'), ('nlms', 'nlms_ho')):
    h = H[fam]
    grid = h['grid']
    assign = h['heldout_assignment']                 # sub -> 参数值
    for s in subs:
        if s not in assign:
            continue
        m = grid[assign[s]]
        for task in ('ERP', 'SSVEP'):
            for sp in SPEEDS:
                k = f'{task}_{sp}_{m}'
                if k in data[s]:
                    data[s][f'{task}_{sp}_{key}'] = data[s][k]
    HO[key] = {'fold_choices': [h['fold1']['chosen'], h['fold2']['chosen']], 'grid': grid}
CCA_LABEL = 'cca r=' + '/'.join(sorted(set(HO['cca_ho']['fold_choices']))) + ' (held-out)'
NLMS_LABEL = 'gated NLMS μ=' + '/'.join(sorted(set(HO['nlms_ho']['fold_choices']))) + ' (held-out)'


def col(task, sp, m, subset=None):
    ss = subs if subset is None else subset
    return {s: data[s][f'{task}_{sp}_{m}'] for s in ss if f'{task}_{sp}_{m}' in data[s]}


def mean_str(task, sp, m, subset=None):
    v = list(col(task, sp, m, subset).values())
    if not v:
        return '—'
    return f'{np.mean(v):.3f}' if task == 'ERP' else f'{np.mean(v):.1f}'


def hodges_lehmann(d):
    n = len(d)
    iu = np.triu_indices(n)
    return float(np.median((d[iu[0]] + d[iu[1]]) / 2))


def paired(task, sp, a, b, subset=None):
    ca, cb = col(task, sp, a, subset), col(task, sp, b, subset)
    ks = sorted(set(ca) & set(cb))
    d = np.array([cb[k] - ca[k] for k in ks])
    if task == 'ERP':
        d = d * 100
    n = len(d)
    if n == 0:
        return None
    nz = d[d != 0]
    p = float(wilcoxon(d, zero_method='zsplit', method='exact').pvalue) if len(nz) else 1.0
    idx = rng.integers(0, n, size=(10000, n))
    boot = d[idx].mean(1)
    boot_hl = np.array([hodges_lehmann(d[i]) for i in idx[:2000]])
    return {'n': n, 'mean': float(d.mean()), 'lo': float(np.percentile(boot, 2.5)), 'hi': float(np.percentile(boot, 97.5)),
            'hl': hodges_lehmann(d), 'hl_lo': float(np.percentile(boot_hl, 2.5)), 'hl_hi': float(np.percentile(boot_hl, 97.5)),
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
    return (f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]; HL {r['hl']:+.2f} [{r['hl_lo']:+.2f}, {r['hl_hi']:+.2f}]; "
            f"{r['up']}/{r['down']}" + (f"/{r['tie']}" if r['tie'] else '') + f"; p={r['p']:.2g}; Holm={r['p_holm']:.2g}")


def family(tests):
    rows = []
    for t, s, a, b, *rest in tests:
        subset = rest[0] if rest else None
        r = paired(t, s, a, b, subset)
        if r:
            r.update({'task': t, 'speed': s, 'a': a, 'b': b})
            rows.append(r)
    if rows:
        for r, pa in zip(rows, holm(np.array([r['p'] for r in rows]))):
            r['p_holm'] = float(pa)
    return rows


out = {'heldout': HO}
L = ['# 论文一 表格 v3(由 make_tables.py 从结果文件生成;差值以 ERP AUC×100 点 / SSVEP 百分点计;HL = Hodges–Lehmann 中位数位移)', '',
     f'留人选参:{CCA_LABEL};{NLMS_LABEL}', '']

# ---- S2 人数表
L += ['## 表 S2 各条件纳入人数(不处理列有值即纳入)', '', '| 条件 | ERP n | SSVEP n | 缺的受试者(ERP / SSVEP) |', '|---|---|---|---|']
incl = {}
for sp in SPEEDS:
    e, v = col('ERP', sp, 'none'), col('SSVEP', sp, 'none')
    me = sorted(set(subs) - set(e)); mv = sorted(set(subs) - set(v))
    incl[sp] = {'ERP': len(e), 'SSVEP': len(v), 'missing_ERP': me, 'missing_SSVEP': mv,
                'ERP_figshare': len([s for s in e if s in FIGSHARE]), 'SSVEP_figshare': len([s for s in v if s in FIGSHARE])}
    L.append(f"| {sp} m/s | {len(e)} | {len(v)} | {', '.join(me) or '—'} / {', '.join(mv) or '—'} |")
out['inclusion'] = incl
L.append('')

# ---- 表 1 均值(主口径:标准顺序 ASR;held-out cca/nlms)
METHODS1 = [('none', 'none'), ('reg', 'reg'), ('reg_b30', 'reg 30 s blocks'), ('asr_std', 'asr'), ('asr10_std', 'asr10'), ('cca_ho', 'cca'), ('icc_w4', 'icc 4 s'),
            ('gait', 'gait'), ('gait+reg', 'gait+reg'), ('nlms_ho', 'gated NLMS (replay)')]
L += ['## 表 1 回顾性流水线各方法均值(ASR = 标准顺序;cca/NLMS = held-out 参数)', '']
for task in ('ERP', 'SSVEP'):
    L += ['| ' + task + ' | ' + ' | '.join(lab for _, lab in METHODS1) + ' |', '|' + '---|' * (len(METHODS1) + 1)]
    for sp in SPEEDS:
        n = len(col(task, sp, 'none'))
        L.append(f'| {sp} m/s ({n}) | ' + ' | '.join(mean_str(task, sp, m) for m, _ in METHODS1) + ' |')
    L.append('')
out['table1'] = {f'{t}_{sp}_{m}': mean_str(t, sp, m) for t in ('ERP', 'SSVEP') for sp in SPEEDS for m, _ in METHODS1 + [('asr', 'asr'), ('asr10', 'asr10'), ('reg_b10', 'reg_b10')]}
out['table1_n'] = {f'{t}_{sp}': len(col(t, sp, 'none')) for t in ('ERP', 'SSVEP') for sp in SPEEDS}

# ---- 表 2 家族统计
STAND = [('ERP', '0.0'), ('SSVEP', '0.0')]
families = {
    'F1 reg vs none': [(t, s, 'none', 'reg') for t, s in COND],
    'F2 gated NLMS vs none': [(t, s, 'none', 'nlms_ho') for t, s in COND],
    'F3 gated NLMS vs reg': [(t, s, 'reg', 'nlms_ho') for t, s in COND],
    'F4 other offline vs none': [(t, s, 'none', m) for m in ('asr_std', 'asr10_std', 'cca_ho', 'icc_w4', 'gait', 'gait+reg') for t, s in COND],
    'F5 standing cost': [(t, s, 'none', m) for m in ('nlms', 'reg', 'reg_b30', 'cca_ho', 'asr_std', 'asr10_std', 'icc_w4') for t, s in STAND],
    'F6 controls vs true reference (reg)': [(t, s, 'reg', m) for m in ('reg_shift10', 'reg_shift', 'reg_shift120', 'reg_surr') for t, s in COND],
    'F7 controls vs true reference (NLMS)': [(t, s, 'nlms_ho', m) for m in ('nlms_shift10', 'nlms_shift', 'nlms_shift120', 'nlms_surr') for t, s in COND],
    'F8 ablation vs 18-ch reg': [(t, s, 'reg', m) for m in ('reg_head', 'reg_ankle') for t, s in COND],
    'F11 blockwise reg vs reg': [(t, s, 'reg', m) for m in ('reg_b10', 'reg_b30') for t, s in COND],
    'F12 gated NLMS vs reg 30 s blocks': [(t, s, 'reg_b30', 'nlms_ho') for t, s in COND],
    'F13 CCA 2x2 variants vs none': [(t, s, 'none', m) for m in ('cca40_nolag', 'icc_w4_lag', 'icc_w20', 'icc_w20_lag') for t, s in COND],
    'F14 rank-safe ASR vs standard-order ASR': [(t, s, b, a) for a, b in (('asr', 'asr_std'), ('asr10', 'asr10_std')) for t, s in COND],
    'F1f reg vs none (figshare only)': [(t, s, 'none', 'reg', FIGSHARE) for t, s in COND],
    'F2f gated NLMS vs none (figshare only)': [(t, s, 'none', 'nlms_ho', FIGSHARE) for t, s in COND],
    'F4f asr10 / cca vs none (figshare only)': [(t, s, 'none', m, FIGSHARE) for m in ('asr10_std', 'cca_ho') for t, s in COND],
    'gait+reg vs reg (direct)': [(t, s, 'reg', 'gait+reg') for t, s in COND],
    'asr10 vs reg (direct)': [(t, s, 'reg', 'asr10_std') for t, s in COND],
}
stats = {}
for fam, tests in families.items():
    rows = family(tests)
    if not rows:
        continue
    stats[fam] = rows
    L += [f'## 表 2 · {fam}', '', '| 条件 | 对比 | 平均差 [95% CI]; HL [CI]; 升/降(/平); p; Holm |', '|---|---|---|']
    for r in rows:
        L.append(f"| {r['task']} {r['speed']} (n={r['n']}) | {r['b']} − {r['a']} | {fmt(r)} |")
    L.append('')
out['stats'] = stats

# ---- 表 3 参数扫描(全部 8 个条件)
L += ['## 表 3 参数扫描(门控 NLMS 步长;cca 阈值 0.2–0.7;icc 窗长;岭系数;分块长度)', '']
sweeps = [('NLMS μ', [('none', 'none'), ('nlms_mu02', '0.02'), ('nlms_gated', '0.05'), ('nlms_mu10', '0.10'), ('nlms_mu20', '0.20')]),
          ('cca r', [('none', 'none'), ('cca20', '0.2'), ('cca', '0.3'), ('cca40', '0.4'), ('cca50', '0.5'), ('cca60', '0.6'), ('cca70', '0.7')]),
          ('icc 窗 × 时滞', [('none', 'none'), ('icc', '2 s'), ('icc_w4', '4 s'), ('icc_w4_lag', '4 s+lags'), ('icc_w20', '20 s'), ('icc_w20_lag', '20 s+lags'), ('cca40_nolag', 'global, no lags'), ('cca40', 'global+lags')]),
          ('reg 岭系数', [('none', 'none'), ('reg_a1', '0.01%'), ('reg', '0.1%'), ('reg_a3', '1%')]),
          ('reg 分块', [('none', 'none'), ('reg_b10', '10 s'), ('reg_b30', '30 s'), ('reg', 'whole segment'), ('nlms_ho', 'gated NLMS')]),
          ('ASR 顺序', [('none', 'none'), ('asr', 'rank-safe 20'), ('asr_std', 'standard 20'), ('asr10', 'rank-safe 10'), ('asr10_std', 'standard 10')])]
sweep_out = {}
for name, grid in sweeps:
    for task in ('ERP', 'SSVEP'):
        L += [f'| {name} · {task} | ' + ' | '.join(lab for _, lab in grid) + ' |', '|' + '---|' * (len(grid) + 1)]
        for sp in SPEEDS:
            L.append(f'| {sp} m/s | ' + ' | '.join(mean_str(task, sp, m) for m, _ in grid) + ' |')
            for m, lab in grid:
                sweep_out[f'{name}|{task}|{sp}|{m}'] = mean_str(task, sp, m)
        L.append('')
out['sweeps'] = sweep_out

# ---- 表 5 对照(均值)
L += ['## 表 5 对照:时移/替代参考与只用惯性解码(均值)', '']
ctrl = [('none', 'none'), ('reg', 'reg 真参考'), ('reg_shift10', 'reg 移 10 s'), ('reg_shift', 'reg 移 30 s'), ('reg_shift120', 'reg 移 120 s'), ('reg_surr', 'reg 相位随机'),
        ('nlms_ho', 'NLMS 真参考'), ('nlms_shift10', 'NLMS 移 10 s'), ('nlms_shift', 'NLMS 移 30 s'), ('nlms_shift120', 'NLMS 移 120 s'), ('nlms_surr', 'NLMS 相位随机')]
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

# ---- 表 6 因果回放(含 figshare 分层与前端敏感性)
def causal_rows(C, subset=None):
    cs = C['per_subject']
    rows = []
    for task in ('ERP', 'SSVEP'):
        for sp in SPEEDS:
            v0 = {s: cs[s][f'{task}_{sp}_none'] for s in cs if f'{task}_{sp}_none' in cs[s] and (subset is None or s in subset)}
            if not v0:
                continue
            vals = {'task': task, 'speed': sp, 'n': len(v0), 'none_mean': float(np.mean(list(v0.values())))}
            for m in ('nlms_c', 'nlms_gated_c'):
                vm = {s: cs[s][f'{task}_{sp}_{m}'] for s in v0 if f'{task}_{sp}_{m}' in cs[s]}
                ks = sorted(set(v0) & set(vm))
                d = np.array([vm[k] - v0[k] for k in ks]) * (100 if task == 'ERP' else 1)
                p = float(wilcoxon(d, zero_method='zsplit', method='exact').pvalue) if np.any(d != 0) else 1.0
                idx = rng.integers(0, len(d), size=(10000, len(d)))
                boot = d[idx].mean(1)
                boot_hl = np.array([hodges_lehmann(d[i]) for i in idx[:2000]])
                vals[m] = {'n': len(d), 'mean': float(d.mean()), 'lo': float(np.percentile(boot, 2.5)), 'hi': float(np.percentile(boot, 97.5)),
                           'hl': hodges_lehmann(d), 'hl_lo': float(np.percentile(boot_hl, 2.5)), 'hl_hi': float(np.percentile(boot_hl, 97.5)),
                           'up': int((d > 0).sum()), 'down': int((d < 0).sum()), 'tie': int((d == 0).sum()), 'p': p, 'task': task, 'speed': sp,
                           'method_mean': float(np.mean(list(vm.values())))}
            rows.append(vals)
    for key in ('nlms_gated_c', 'nlms_c'):
        for v, pa in zip(rows, holm(np.array([v[key]['p'] for v in rows]))):
            v[key]['p_holm'] = float(pa)
    return rows


cz = os.path.join(RC, 'results_thr100_mu0.05.json')
if os.path.exists(cz):
    C = json.load(open(cz))
    crows = causal_rows(C)
    L += ['## 表 6 因果回放(零阶保持、门控阈值 100 = 留出两折所选;0.5 Hz 因果高通)', '', '| 条件 | none | NLMS | gated NLMS | gated − none | NLMS − none |', '|---|---|---|---|---|---|']
    for v in crows:
        f = (lambda x: f'{x:.3f}') if v['task'] == 'ERP' else (lambda x: f'{x:.1f}')
        L.append(f"| {v['task']} {v['speed']} (n={v['n']}) | {f(v['none_mean'])} | {f(v['nlms_c']['method_mean'])} | {f(v['nlms_gated_c']['method_mean'])} | {fmt(v['nlms_gated_c'])} | {fmt(v['nlms_c'])} |")
    g = C['gate']
    L += ['', f"门控逐秒决策(以段速度为真值):TP {g['per_second_decisions']['TP']:.0f}, FN {g['per_second_decisions']['FN']:.0f}, FP {g['per_second_decisions']['FP']:.0f}, TN {g['per_second_decisions']['TN']:.0f};灵敏度 {g['sensitivity']:.3f},特异度 {g['specificity']:.3f}", '']
    per = {}
    for seg in C['gate_segments']:
        per.setdefault(seg['sub'], {'mov_on': [], 'stand_on': []})
        (per[seg['sub']]['mov_on'] if seg['moving'] else per[seg['sub']]['stand_on']).append(seg['gate_on_frac'])
    L += ['逐人门控:行走段门开比例均值 / 站立段门开比例均值', '', '| 受试者 | 行走段开门 | 站立段开门 |', '|---|---|---|']
    for s in sorted(per):
        mo = np.mean(per[s]['mov_on']) if per[s]['mov_on'] else float('nan')
        so = np.mean(per[s]['stand_on']) if per[s]['stand_on'] else float('nan')
        L.append(f'| {s} | {mo:.2f} | {so:.2f} |')
    # 按 release 的门控表现
    rel = {}
    for name, sel in (('figshare', FIGSHARE), ('OSF', OSF)):
        segs = [x for x in C['gate_segments'] if x['sub'] in sel]
        st = [x['gate_on_frac'] for x in segs if not x['moving']]
        mv = [x['gate_on_frac'] for x in segs if x['moving']]
        rel[name] = {'standing_on_mean': float(np.mean(st)), 'standing_on_max': float(np.max(st)), 'n_standing_segments': len(st),
                     'moving_on_mean': float(np.mean(mv)), 'moving_on_min': float(np.min(mv)), 'n_moving_segments': len(mv)}
    L += ['', '按 release 的门控:' + '; '.join(f"{k}: 站立段开门均值 {v['standing_on_mean']:.3f}(最大 {v['standing_on_max']:.2f}),行走段开门均值 {v['moving_on_mean']:.3f}(最小 {v['moving_on_min']:.2f})" for k, v in rel.items()), '']
    ts = C.get('transition_summary', {})
    L += ['过渡实验(站立段+快走段相接):' + '; '.join(f"{k}={v['mean']:.3f}" for k, v in ts.items()), '']
    # 过渡实验走路段配对变化
    tr = C.get('transition', {})
    dw = np.array([t['ssvep_walking_gated'] - t['ssvep_walking_none'] for t in tr.values()])
    idx = rng.integers(0, len(dw), size=(10000, len(dw)))
    trans_walk = {'n': len(dw), 'mean': float(dw.mean()), 'lo': float(np.percentile(dw[idx].mean(1), 2.5)), 'hi': float(np.percentile(dw[idx].mean(1), 97.5)),
                  'up': int((dw > 0).sum()), 'down': int((dw < 0).sum()), 'tie': int((dw == 0).sum()),
                  'p': float(wilcoxon(dw, zero_method='zsplit', method='exact').pvalue) if np.any(dw != 0) else 1.0}
    L += [f"过渡实验走路段 SSVEP 配对变化:{trans_walk['mean']:+.2f} [{trans_walk['lo']:+.2f}, {trans_walk['hi']:+.2f}],{trans_walk['up']}/{trans_walk['down']}/{trans_walk['tie']},p={trans_walk['p']:.2g}", '']
    # figshare 分层
    crows_f = causal_rows(C, FIGSHARE)
    L += ['### 表 6f 因果回放,figshare 18 人', '', '| 条件 | none | gated | gated − none |', '|---|---|---|---|']
    for v in crows_f:
        f = (lambda x: f'{x:.3f}') if v['task'] == 'ERP' else (lambda x: f'{x:.1f}')
        L.append(f"| {v['task']} {v['speed']} (n={v['n']}) | {f(v['none_mean'])} | {f(v['nlms_gated_c']['method_mean'])} | {fmt(v['nlms_gated_c'])} |")
    L.append('')
    out['causal'] = {'rows': crows, 'rows_figshare': crows_f, 'gate': g, 'gate_by_release': rel, 'transition': ts, 'transition_walk_paired': trans_walk}
    # 前端敏感性:0.1 Hz 因果高通
    cz2 = os.path.join(RC, 'results_thr100_mu0.05_hp0.1.json')
    if os.path.exists(cz2):
        C2 = json.load(open(cz2))
        crows2 = causal_rows(C2)
        L += ['### 表 6s 因果回放前端敏感性:0.1 Hz 因果高通(其余相同)', '', '| 条件 | none | NLMS | gated | gated − none |', '|---|---|---|---|---|']
        for v in crows2:
            f = (lambda x: f'{x:.3f}') if v['task'] == 'ERP' else (lambda x: f'{x:.1f}')
            L.append(f"| {v['task']} {v['speed']} (n={v['n']}) | {f(v['none_mean'])} | {f(v['nlms_c']['method_mean'])} | {f(v['nlms_gated_c']['method_mean'])} | {fmt(v['nlms_gated_c'])} |")
        # 前端之间 none 的配对差(0.1 vs 0.5 Hz)
        fe = []
        for task in ('ERP', 'SSVEP'):
            for sp in SPEEDS:
                a = {s: C['per_subject'][s][f'{task}_{sp}_none'] for s in C['per_subject'] if f'{task}_{sp}_none' in C['per_subject'][s]}
                b = {s: C2['per_subject'][s][f'{task}_{sp}_none'] for s in C2['per_subject'] if f'{task}_{sp}_none' in C2['per_subject'][s]}
                ks = sorted(set(a) & set(b))
                d = np.array([b[k] - a[k] for k in ks]) * (100 if task == 'ERP' else 1)
                fe.append({'task': task, 'speed': sp, 'n': len(d), 'mean': float(d.mean()), 'up': int((d > 0).sum()), 'down': int((d < 0).sum())})
        L += ['', '0.1 Hz − 0.5 Hz 前端(不处理)的配对差:' + '; '.join(f"{x['task']} {x['speed']}: {x['mean']:+.2f} ({x['up']}/{x['down']})" for x in fe), '']
        out['causal_hp01'] = {'rows': crows2, 'frontend_diff': fe, 'gate': C2['gate']}
    # 零相位 vs 因果前端(不处理)配对差
    zp = []
    for task in ('ERP', 'SSVEP'):
        for sp in SPEEDS:
            a = col(task, sp, 'none')
            b = {s: C['per_subject'][s][f'{task}_{sp}_none'] for s in C['per_subject'] if f'{task}_{sp}_none' in C['per_subject'][s]}
            ks = sorted(set(a) & set(b))
            d = np.array([b[k] - a[k] for k in ks]) * (100 if task == 'ERP' else 1)
            zp.append({'task': task, 'speed': sp, 'n': len(d), 'mean': float(d.mean()), 'up': int((d > 0).sum()), 'down': int((d < 0).sum())})
    L += ['因果 0.5 Hz − 零相位前端(不处理)的配对差:' + '; '.join(f"{x['task']} {x['speed']}: {x['mean']:+.2f} ({x['up']}/{x['down']})" for x in zp), '']
    out['causal_vs_zerophase'] = zp

# ---- 表 7 ASR 三种口径
old = json.load(open(os.path.join(R, 'results_24.json')))['per_subject']
new = json.load(open(os.path.join(R, 'results_24_asrfix_icc.json')))['per_subject']
L += ['## 表 7 ASR(阈值 20)三种口径,ERP 1.6 m/s 均值:CAR 后直接 ASR / CAR 后 rank-safe / 标准顺序', '', '| 组 | n | 不处理 | CAR→ASR 直接 | CAR→ASR rank-safe | 标准顺序 ASR | 标准顺序 ASR10 |', '|---|---|---|---|---|---|---|']
asr_tab = {}
for grp, sel in (('figshare 18', FIGSHARE), ('OSF 6', OSF)):
    ks = [s for s in sel if 'ERP_1.6_asr' in old.get(s, {}) and 'ERP_1.6_asr' in new.get(s, {}) and 'ERP_1.6_asr_std' in data.get(s, {})]
    row = {'n': len(ks), 'none': float(np.mean([old[s]['ERP_1.6_none'] for s in ks])), 'direct': (float(np.mean([old[s]['ERP_1.6_asr'] for s in ks])) if grp.startswith('figshare') else None),
           'ranksafe': float(np.mean([new[s]['ERP_1.6_asr'] for s in ks])), 'std': float(np.mean([data[s]['ERP_1.6_asr_std'] for s in ks])),
           'std10': float(np.mean([data[s]['ERP_1.6_asr10_std'] for s in ks]))}
    asr_tab[grp] = row
    L.append(f"| {grp} | {row['n']} | {row['none']:.3f} | {'—' if row['direct'] is None else ('%.3f' % row['direct'])} | {row['ranksafe']:.3f} | {row['std']:.3f} | {row['std10']:.3f} |")
asr_std_file = os.path.join(R, 'results_24_asr_std.json')
if os.path.exists(asr_std_file):
    er = json.load(open(asr_std_file)).get('energy_ratio', {})
    allr = [v for r in er.values() for v in r.values()]
    if allr:
        asr_tab['energy_ratio'] = {'min': float(min(allr)), 'median': float(np.median(allr)), 'max': float(max(allr)), 'n_segments': len(allr)}
        L += ['', f"标准顺序 ASR 输出/输入能量比:最小 {min(allr):.3f},中位 {np.median(allr):.3f},最大 {max(allr):.3f}({len(allr)} 段)"]
out['asr_orders'] = asr_tab
L.append('')

# ---- 表 8 耗时
tp = os.path.join(R, 'timing.json')
if os.path.exists(tp):
    t = json.load(open(tp))
    L += ['## 表 8 单核处理每秒数据耗时(ms)', '', '| ' + ' | '.join(t['ms_per_s']) + ' |', '|' + '---|' * len(t['ms_per_s']), '| ' + ' | '.join(f'{v:.1f}' for v in t['ms_per_s'].values()) + ' |', '']
    out['timing'] = t

# ---- 表 9 SSVEP 信号探针(走路损失是信号弱了还是分类器受扰)
pr = os.path.join(R, 'ssvep_signal_probe.json')
if os.path.exists(pr):
    P = json.load(open(pr))
    L += ['## 表 9 SSVEP 信号探针:正确参考 CCA 相关 / 最强错误参考相关 / margin(均值)', '', '| 速度 | n | none 正确 | none 错误 | none margin | reg 正确 | reg margin | NLMS 正确 | NLMS margin |', '|---|---|---|---|---|---|---|---|---|']
    for sp in SPEEDS:
        g = lambda k: P['summary'][k]['mean'] if k in P['summary'] else float('nan')
        n = P['summary'].get(f'{sp}_none_margin', {}).get('n', 0)
        L.append(f"| {sp} m/s | {n} | {g(f'{sp}_none_corr_true'):.3f} | {g(f'{sp}_none_corr_wrong'):.3f} | {g(f'{sp}_none_margin'):.3f} | {g(f'{sp}_reg_corr_true'):.3f} | {g(f'{sp}_reg_margin'):.3f} | {g(f'{sp}_nlms_gated_corr_true'):.3f} | {g(f'{sp}_nlms_gated_margin'):.3f} |")
    L += ['', '配对(各速度 − 站立,不处理):' + '; '.join(f"{k}: {v['mean']:+.3f} ({v['up']}/{v['down']})" for k, v in P['paired_vs_standing_none'].items()), '']
    out['ssvep_probe'] = {'summary': P['summary'], 'paired': P['paired_vs_standing_none']}
# ---- 表 10 跨 release 一致性
rc = os.path.join(ROOT, 'results', 'release_concordance.json')
if os.path.exists(rc):
    RCJ = json.load(open(rc))
    L += ['## 表 10 跨 release 一致性(同一录音:OSF 100 Hz − figshare 500 Hz)', '', '| 方法 | 任务 | 速度 | n | 均值差 | 最大|差| | 方法效应差均值 | 方法效应差最大 |', '|---|---|---|---|---|---|---|---|']
    for r in RCJ['rows']:
        ed = '' if r['effect_diff_mean'] is None else f"{r['effect_diff_mean']:+.2f}"
        em = '' if r['effect_max_abs'] is None else f"{r['effect_max_abs']:.2f}"
        L.append(f"| {r['method']} | {r['task']} | {r['speed']} | {r['n']} | {r['osf_minus_figshare_mean']:+.2f} | {r['max_abs_diff']:.2f} | {ed} | {em} |")
    L.append('')
    out['release_concordance'] = RCJ['rows']
json.dump(out, open(os.path.join(ROOT, 'results', 'tables_v3.json'), 'w'), indent=1, ensure_ascii=False)
open(os.path.join(ROOT, 'docs', 'tables_v3.md'), 'w').write('\n'.join(L))
print('\n'.join(L))
