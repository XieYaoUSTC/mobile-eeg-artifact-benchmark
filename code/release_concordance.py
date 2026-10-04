#!/usr/bin/env python3
"""跨 release 一致性(补充材料 S16):同一录音经 figshare(500 Hz 原始)与 OSF(100 Hz,已坏道插值)两条 release 的解码结果对比。
重复对(事件序列比对确认):sub-19=s07, sub-20=s13, sub-21=s15, sub-22=s16, sub-23=s17, sub-24=s18。
数据来源:results/p1_clean_eval/_v3_with_duplicates/(并入新被试前归档的结果,含这六个 OSF 重复被试);若尚未归档则读当前文件。
输出 results/release_concordance.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'p1_clean_eval')
ARCH = os.path.join(R, '_v3_with_duplicates')
SRC = ARCH if os.path.isdir(ARCH) else R
PAIRS = {'sub-19': 's07', 'sub-20': 's13', 'sub-21': 's15', 'sub-22': 's16', 'sub-23': 's17', 'sub-24': 's18'}
data = {}
for f in ['results_24.json', 'results_24_gated.json', 'results_24_asrfix_icc.json', 'results_24_asr_std.json']:
    pth = os.path.join(SRC, f)
    if not os.path.exists(pth):
        continue
    for s, d in json.load(open(pth))['per_subject'].items():
        if f == 'results_24_asr_std.json':
            d = {k: v for k, v in d.items() if not k.endswith('_none')}
        data.setdefault(s, {}).update(d)
out = {'pairs': PAIRS, 'source': SRC, 'rows': []}
for m in ['none', 'reg', 'nlms_gated', 'asr10_std', 'asr10']:
    for task in ['ERP', 'SSVEP']:
        for sp in ['0.0', '0.8', '1.6']:
            d, eff = [], []
            for o, s in PAIRS.items():
                k, k0 = f'{task}_{sp}_{m}', f'{task}_{sp}_none'
                if k in data.get(o, {}) and k in data.get(s, {}):
                    sc = 100 if task == 'ERP' else 1
                    d.append((data[o][k] - data[s][k]) * sc)
                    if m != 'none' and k0 in data[o] and k0 in data[s]:
                        eff.append(((data[o][k] - data[o][k0]) - (data[s][k] - data[s][k0])) * sc)
            if d:
                out['rows'].append({'method': m, 'task': task, 'speed': sp, 'n': len(d), 'osf_minus_figshare_mean': float(np.mean(d)),
                                    'max_abs_diff': float(np.max(np.abs(d))), 'per_pair': [round(x, 2) for x in d],
                                    'effect_diff_mean': float(np.mean(eff)) if eff else None, 'effect_max_abs': float(np.max(np.abs(eff))) if eff else None})
json.dump(out, open(os.path.join(ROOT, 'results', 'release_concordance.json'), 'w'), indent=1)
for r in out['rows']:
    print(f"{r['method']:10s} {r['task']:5s} {r['speed']} n={r['n']} OSF−figshare mean {r['osf_minus_figshare_mean']:+.2f} max|d| {r['max_abs_diff']:.2f}" + (f"  effect diff mean {r['effect_diff_mean']:+.2f} max {r['effect_max_abs']:.2f}" if r['effect_diff_mean'] is not None else ''))
