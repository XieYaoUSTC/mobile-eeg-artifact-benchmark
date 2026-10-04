#!/usr/bin/env python3
"""腕部外骨骼数据集(EEG-EMG-WristMotion, Sci Data 2026)基线:运动期节律下降(ERD)。

原作者的技术验证是逐人的 ERD/ERS 图与功率谱图,没有单一数字。这里给一个可复算的数:
  每次试验 10 s = 注视 3 s + 目标预览 2 s + 运动 2.5 s + 回中 2.5 s
  参考期 = 目标预览 3–5 s(手不动);运动期 = 5–7.5 s
  ERD% = (运动期功率 − 参考期功率) / 参考期功率 × 100,负值表示节律下降
  频段:mu 8–13 Hz、beta 13–30 Hz;通道 C3(右手对侧)与 C4
  剔除:任一脑电通道峰峰值超过 100 微伏的试次(原作者的标准)
同时核对外骨骼编码器:运动期腕关节角度的变化幅度应明显大于参考期。
只统计任务段;每人首尾两个文件是静息段(事件戳照打,但手不动),按角度是否变化自动识别并排除。
"""
import glob
import json
import os
import zipfile

import mne
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

mne.set_log_level('ERROR')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'wrist_exo')
OUT = os.path.join(ROOT, 'results', 'wrist_exo_baseline')
EEG = ['C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CPz', 'CP4']       # EDF 里的 Ch1–Ch8
BANDS = {'mu': (8, 13), 'beta': (13, 30)}
TRIAL, REF, MOV = 10.0, (3.0, 5.0), (5.0, 7.5)                # 各段起点相对文件开头另有约 0.04 s 的固定偏移,可忽略


def band_power(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    return np.abs(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)) ** 2


def run_file(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    fs = raw.info['sfreq']
    d = raw.get_data()
    names = raw.ch_names
    eeg = d[:8] * (1e6 if np.nanmax(np.abs(d[:8])) < 1 else 1.0)   # 统一成微伏
    eeg = eeg - eeg.mean(0, keepdims=True)                      # 平均参考(8 个通道)
    # 腕关节角度取自单独的 motion.tsv(编码器计数,零位约 32768)。EDF 里的 X/Y 通道超出 16 位范围被截断,不能用。
    mfile = path.replace('/eeg/', '/motion/').replace('_task-WristPointingTask_run', '_task-WristPointingTask_tracksys-BiomechWrist_run').replace('_eeg.edf', '_motion.tsv')
    xy = np.loadtxt(mfile, delimiter='\t').T[:, :d.shape[1]]
    if max(len(np.unique(xy[0])), len(np.unique(xy[1]))) < 50:
        return None, 0, 0                                       # 静息段(每人首尾各一个文件):手没动,不算任务试次
    xy = np.where(xy < 1000, np.nan, xy)                        # 文件开头有若干 0 值
    n = int(TRIAL * fs)
    n_trials = int(d.shape[1] // n)
    sos = butter(4, [1, 40], btype='band', fs=fs, output='sos')
    bb = sosfiltfilt(sos, eeg, axis=-1)
    pw = {k: band_power(eeg, fs, *v) for k, v in BANDS.items()}
    r0, r1, m0, m1 = (int(t * fs) for t in (*REF, *MOV))
    out = {f'{b}_{c}': [] for b in BANDS for c in ('C3', 'C4')}
    out.update({'move_range': [], 'ref_range': []})
    kept = 0
    for i in range(n_trials):
        s = i * n
        seg = bb[:, s + r0:s + m1]
        if np.ptp(seg, axis=1).max() > 100:
            continue
        kept += 1
        for b in BANDS:
            for c in ('C3', 'C4'):
                ci = EEG.index(c)
                pr = pw[b][ci, s + r0:s + r1].mean()
                pm = pw[b][ci, s + m0:s + m1].mean()
                out[f'{b}_{c}'].append((pm - pr) / pr * 100)
        out['move_range'].append(float(np.nanmax(np.nanmax(xy[:, s + m0:s + m1], axis=1) - np.nanmin(xy[:, s + m0:s + m1], axis=1))))
        out['ref_range'].append(float(np.nanmax(np.nanmax(xy[:, s + r0:s + r1], axis=1) - np.nanmin(xy[:, s + r0:s + r1], axis=1))))
    return out, n_trials, kept


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for z in sorted(glob.glob(os.path.join(DATA, 'sub-*.zip'))):
        sub = os.path.basename(z)[:-4]
        if not os.path.isdir(os.path.join(DATA, sub)):
            zipfile.ZipFile(z).extractall(DATA)
        agg, tot, kept = {}, 0, 0
        for f in sorted(glob.glob(os.path.join(DATA, sub, 'eeg', '*_eeg.edf'))):
            try:
                o, n, k = run_file(f)
            except Exception as e:
                print(sub, os.path.basename(f), 'ERR', e, flush=True)
                continue
            if o is None:
                continue
            tot += n
            kept += k
            for key, v in o.items():
                agg.setdefault(key, []).extend(v)
        if kept == 0:
            print(sub, 'no usable trials', flush=True)
            continue
        r = {'sub': sub, 'trials': tot, 'kept': kept}
        r.update({k: float(np.median(v)) for k, v in agg.items()})
        rows.append(r)
        print(sub, f"保留 {kept}/{tot}  mu_C3={r['mu_C3']:.1f}%  beta_C3={r['beta_C3']:.1f}%  角度变化 运动期 {r['move_range']:.3f} / 参考期 {r['ref_range']:.3f}", flush=True)
    summ = {'n_subjects': len(rows)}
    for k in ['mu_C3', 'beta_C3', 'mu_C4', 'beta_C4']:
        v = np.array([r[k] for r in rows])
        summ[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'frac_negative': float((v < 0).mean())}
    summ['kept_ratio'] = float(sum(r['kept'] for r in rows) / sum(r['trials'] for r in rows))
    json.dump({'rows': rows, 'summary': summ}, open(os.path.join(OUT, 'baseline.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 汇总 ===')
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
