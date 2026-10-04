#!/usr/bin/env python3
"""审稿后数据修正:OSF 的 sub-19…24 与 figshare 的 s07/s13/s15/s16/s17/s18 是同一批录音(事件序列 100% 吻合),
v3 的"24 人"实际是 18 人 + 6 个重复。本脚本把结果文件里的重复 OSF 被试剔除,并把真正新增的 OSF 被试(单独用 SUBS=… 跑出的 *_new6 文件)并入。
用法: merge_new_subjects.py <dup1,dup2,…> <new1,new2,…>
原文件先归档到 results/p1_clean_eval/_v3_with_duplicates/ 与 results/p1_causal_replay/_v3_with_duplicates/。
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'p1_clean_eval')
RC = os.path.join(ROOT, 'results', 'p1_causal_replay')
DUP = sys.argv[1].split(',')
NEW = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else []
SPEEDS = ['0.0', '0.8', '1.6', '2.0']


def archive(path):
    d = os.path.join(os.path.dirname(path), '_v3_with_duplicates')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy2(path, dst)


def summarize(rows, methods):
    summ = {}
    for task in ('ERP', 'SSVEP'):
        for sp in SPEEDS:
            for m in methods:
                v = [r[f'{task}_{sp}_{m}'] for r in rows.values() if f'{task}_{sp}_{m}' in r]
                if v:
                    summ[f'{task}_{sp}_{m}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0, 'n': len(v)}
    return summ


# ---- 回顾性结果文件
for f in sorted(glob.glob(os.path.join(R, 'results_24*.json'))):
    if f.endswith('_new6.json'):
        continue
    archive(f)
    D = json.load(open(f))
    rows = {s: r for s, r in D['per_subject'].items() if s not in DUP}
    nf = f.replace('.json', '_new6.json')
    added = 0
    if os.path.exists(nf):
        for s, r in json.load(open(nf))['per_subject'].items():
            if s in NEW:
                rows[s] = r
                added += 1
    methods = sorted({k.split('_', 2)[2] for r in rows.values() for k in r})
    D['per_subject'] = rows
    D['summary'] = summarize(rows, methods)
    if 'energy_ratio' in D:
        er = {s: v for s, v in D['energy_ratio'].items() if s not in DUP}
        if os.path.exists(nf):
            er.update({s: v for s, v in json.load(open(nf)).get('energy_ratio', {}).items() if s in NEW})
        D['energy_ratio'] = er
    D['note_v4'] = f'duplicate OSF participants {DUP} removed; new OSF participants {NEW} merged from {os.path.basename(nf)}'
    json.dump(D, open(f, 'w'), indent=1)
    print(os.path.basename(f), 'subjects', len(rows), 'added', added)

# ---- 只用惯性解码
f = os.path.join(R, 'imu_only_decoding.json')
if os.path.exists(f):
    archive(f)
    D = json.load(open(f))
    rows = {s: r for s, r in D['per_subject'].items() if s not in DUP}
    nf = f.replace('.json', '_new6.json')
    if os.path.exists(nf):
        rows.update({s: r for s, r in json.load(open(nf))['per_subject'].items() if s in NEW})
    summ = {}
    keys = sorted({k for r in rows.values() for k in r})
    for k in keys:
        v = [r[k] for r in rows.values() if k in r]
        summ[k] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0, 'n': len(v)}
    D['per_subject'], D['summary'] = rows, summ
    json.dump(D, open(f, 'w'), indent=1)
    print('imu_only_decoding.json subjects', len(rows))

# ---- 因果回放
for f in sorted(glob.glob(os.path.join(RC, 'results_thr*_mu*.json'))):
    if f.endswith('_new6.json'):
        continue
    archive(f)
    D = json.load(open(f))
    rows = {s: r for s, r in D['per_subject'].items() if s not in DUP}
    gates = [g for g in D['gate_segments'] if g['sub'] not in DUP]
    trans = {s: t for s, t in D.get('transition', {}).items() if s not in DUP}
    nf = f.replace('.json', '_new6.json')
    if os.path.exists(nf):
        N = json.load(open(nf))
        rows.update({s: r for s, r in N['per_subject'].items() if s in NEW})
        gates += [g for g in N['gate_segments'] if g['sub'] in NEW]
        trans.update({s: t for s, t in N.get('transition', {}).items() if s in NEW})
    D['per_subject'] = rows
    D['summary'] = summarize(rows, ['none', 'nlms_c', 'nlms_gated_c'])
    D['gate_segments'] = gates
    tp = sum(g['gate_on_frac'] * g['n_decisions'] for g in gates if g['moving'])
    fn = sum((1 - g['gate_on_frac']) * g['n_decisions'] for g in gates if g['moving'])
    fp = sum(g['gate_on_frac'] * g['n_decisions'] for g in gates if not g['moving'])
    tn = sum((1 - g['gate_on_frac']) * g['n_decisions'] for g in gates if not g['moving'])
    D['gate'] = {'per_second_decisions': {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn}, 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
                 'segments_moving_with_gate_on_frac_lt_0.9': [f"{g['sub']}/{g['task']}/{g['speed']}:{g['gate_on_frac']:.2f}" for g in gates if g['moving'] and g['gate_on_frac'] < 0.9],
                 'segments_standing_with_gate_on_frac_gt_0.1': [f"{g['sub']}/{g['task']}/{g['speed']}:{g['gate_on_frac']:.2f}" for g in gates if not g['moving'] and g['gate_on_frac'] > 0.1]}
    D['transition'] = trans
    tsum = {}
    if trans:
        for k in next(iter(trans.values())):
            v = np.array([t[k] for t in trans.values()], float)
            tsum[k] = {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)), 'n': int(np.isfinite(v).sum())}
    D['transition_summary'] = tsum
    D['note_v4'] = f'duplicate OSF participants {DUP} removed; new OSF participants {NEW} merged'
    json.dump(D, open(f, 'w'), indent=1)
    print(os.path.basename(f), 'subjects', len(rows), 'gate segments', len(gates), 'transitions', len(trans))
print('merge done')
