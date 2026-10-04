#!/usr/bin/env python3
"""核对两份 release 的被试对应关系:只抓 OSF 24 人的 events.tsv(小文件),与 figshare 18 人的事件时刻序列(100 Hz 采样点)比对。
判据:同任务同速度的前 50 个事件时刻逐个相差 ≤ 2 个采样点的比例;≥ 0.95 视为同一录音。
输出 results/osf_figshare_mapping.json:每个 OSF 被试的最佳匹配、匹配分数、是否为 figshare 之外的新被试。
前置:code/download_osf_mobile_bci.py list(清单含全部 24 人)。
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSF = os.path.join(ROOT, 'data', 'mobile_bci_osf')
MAN = json.load(open(os.path.join(OSF, '_manifest.json')))
FIG = json.load(open(os.path.join(ROOT, 'results', 'figshare_event_onsets_100hz.json')))   # 由 scratch 脚本缓存:key = sub|task|speed
SES = {'01': 'tr', '02': '0.0', '03': '0.8', '04': '1.6', '05': '2.0'}

# 1) 抓全部 events.tsv(已有的跳过)
ev_items = [it for it in MAN if it['path'].endswith('_events.tsv')]
for it in ev_items:
    dst = os.path.join(OSF, '_events_all', it['path'])
    if os.path.exists(dst) and os.path.getsize(dst) == it['size']:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(['curl', '-sS', '-L', '--retry', '3', '-o', dst, it['url']], check=False)
print('events files:', len(ev_items))

# 2) 比对
osf_subs = sorted({it['path'].split('/')[0] for it in ev_items})
fig_subs = sorted({k.split('|')[0] for k in FIG})
out = {}
for o in osf_subs:
    best = []
    for s in fig_subs:
        sc = []
        for it in ev_items:
            if not it['path'].startswith(o + '/'):
                continue
            parts = os.path.basename(it['path']).split('_')
            ses = parts[1].replace('ses-', ''); task = parts[2].replace('task-', '')
            sp = SES.get(ses)
            key = f'{s}|{task}|{sp}'
            if sp is None or key not in FIG:
                continue
            f = os.path.join(OSF, '_events_all', it['path'])
            if not os.path.exists(f):
                continue
            b = pd.read_csv(f, sep='\t')['onset'].to_numpy().astype(int)
            a = np.array(FIG[key])
            n = min(len(a), len(b), 50)
            if n < 5:
                continue
            sc.append(float(np.mean(np.abs(a[:n] - b[:n]) <= 2)))
        if sc:
            best.append((float(np.mean(sc)), s, len(sc)))
    best.sort(reverse=True)
    out[o] = {'best': best[:3], 'match': best[0][1] if best and best[0][0] >= 0.95 else None}
    print(o, [(round(x[0], 2), x[1], x[2]) for x in best[:3]], '->', out[o]['match'], flush=True)
new = [o for o in osf_subs if out[o]['match'] is None]
mapped = {o: out[o]['match'] for o in osf_subs if out[o]['match']}
print('new OSF subjects (not in figshare):', new)
print('figshare covered:', sorted(set(mapped.values())), 'missing figshare:', sorted(set(fig_subs) - set(mapped.values())))
json.dump({'per_osf': out, 'new_osf_subjects': new, 'osf_to_figshare': mapped}, open(os.path.join(ROOT, 'results', 'osf_figshare_mapping.json'), 'w'), indent=1)
