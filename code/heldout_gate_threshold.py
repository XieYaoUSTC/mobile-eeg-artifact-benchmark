#!/usr/bin/env python3
"""因果门控阈值 θ 的留人选参(与 heldout_param_selection.py 同一分半、同一种子 20261004)。
目标:门控逐秒决策对段速度标签的平衡准确率(灵敏度与特异度的均值),在一半人上选 θ∈{50,100,200},另一半上评,再交换。
数据来自 results/p1_causal_replay/results_thr{50,100,200}_mu0.05.json 的 gate_segments。
另给:每人按"另一半选出的 θ"得到的留出解码均值(heldout_decoding)与每人所用阈值(chosen_per_subject)。
输出 results/p1_causal_replay/heldout_gate_threshold.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC = os.path.join(ROOT, 'results', 'p1_causal_replay')
THR = ['50', '100', '200']
RES = {t: json.load(open(os.path.join(RC, f'results_thr{t}_mu0.05.json'))) for t in THR}
subs = sorted(RES['100']['per_subject'])
rng = np.random.default_rng(20261004)
perm = rng.permutation(len(subs))
halves = [sorted(subs[i] for i in perm[:len(subs) // 2]), sorted(subs[i] for i in perm[len(subs) // 2:])]


def gate_perf(thr, subset):
    segs = [g for g in RES[thr]['gate_segments'] if g['sub'] in subset]
    tp = sum(g['gate_on_frac'] * g['n_decisions'] for g in segs if g['moving'])
    fn = sum((1 - g['gate_on_frac']) * g['n_decisions'] for g in segs if g['moving'])
    fp = sum(g['gate_on_frac'] * g['n_decisions'] for g in segs if not g['moving'])
    tn = sum((1 - g['gate_on_frac']) * g['n_decisions'] for g in segs if not g['moving'])
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return (sens + spec) / 2, sens, spec


out = {'halves': halves}
chosen = {}
for i, (sel, test) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
    scores = {t: gate_perf(t, sel)[0] for t in THR}
    best = max(scores, key=scores.get)
    ba, se, sp = gate_perf(best, test)
    out[f'fold{i + 1}'] = {'scores_on_selection_half': scores, 'chosen': best, 'heldout_balanced_acc': ba, 'heldout_sens': se, 'heldout_spec': sp}
    for s in test:
        chosen[s] = best
    print(f'fold{i + 1}: scores {scores} chosen {best}; held-out BA {ba:.3f} sens {se:.3f} spec {sp:.3f}')
hd = {}
for task in ('ERP', 'SSVEP'):
    for sp in ('0.0', '0.8', '1.6', '2.0'):
        for m in ('none', 'nlms_c', 'nlms_gated_c'):
            v = [RES[chosen[s]]['per_subject'][s][f'{task}_{sp}_{m}'] for s in subs if f'{task}_{sp}_{m}' in RES[chosen[s]]['per_subject'][s]]
            if v:
                hd[f'{task}_{sp}_{m}'] = {'mean': float(np.mean(v)), 'n': len(v)}
out['heldout_decoding'] = hd
out['chosen_per_subject'] = chosen
pooled = gate_perf('100', subs)
out['pooled_thr100'] = {'balanced_acc': pooled[0], 'sens': pooled[1], 'spec': pooled[2]}
json.dump(out, open(os.path.join(RC, 'heldout_gate_threshold.json'), 'w'), indent=1)
print('pooled θ=100:', pooled)
