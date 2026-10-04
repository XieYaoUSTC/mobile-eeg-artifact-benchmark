#!/usr/bin/env python3
"""Mobile BCI:从连续原始数据 raw_x 自己分段(0–5 s,与原作者代码一致)重算 SSVEP 的 CCA 准确率。
数据集自带的 preprocess_x 只有 4 s;这里核对「窗口长度」能否解释与原文的差距。
处理:0.5 Hz 高通(5 阶巴特沃斯)、32 导平均参考、降到 100 Hz;不做去眼电与坏道插值。"""
import glob, json, os, re, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, resample_poly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_mobile_bci import DATA, OUT, SPEEDS, SSVEP_CH, SSVEP_DIV, cca_mean_corr, complete

def run(path, win):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    fs = int(d['raw_fs']); clab = [str(c) for c in d['raw_clab']]
    x = d['raw_x'].astype(float)[:, :32].T                      # 32 导头皮
    x = sosfiltfilt(butter(5, 0.5, btype='high', fs=fs, output='sos'), x, axis=-1)
    x = x - x.mean(0, keepdims=True)
    x = resample_poly(x, 100, fs, axis=-1)
    idx = [clab.index(c) for c in SSVEP_CH]
    on = (np.asarray(d['event'].time, float) / 1000 * 100).astype(int)      # 事件时间是毫秒
    y = np.argmax(d['event'].y, 0)
    n = int(win * 100); tt = np.arange(1, n + 1) / 100
    refs = [np.stack([np.sin(2*np.pi*60/k*tt), np.cos(2*np.pi*60/k*tt), np.sin(4*np.pi*60/k*tt), np.cos(4*np.pi*60/k*tt)], 1) for k in SSVEP_DIV]
    pred = [int(np.argmax([cca_mean_corr(x[idx, o:o+n].T, r) for r in refs])) for o in on if o + n <= x.shape[1]]
    return float(np.mean(np.array(pred) == y[:len(pred)]) * 100)

res = {}
for win in (4, 5):
    for sp in SPEEDS:
        v = [run(f, win) for f in sorted(glob.glob(os.path.join(DATA, f's*_scalp_SSVEP_{sp}.mat'))) if complete(f)]
        res[f'{win}s_{sp}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)}
        print(f'窗口 {win} s, {sp} m/s: {np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} (n={len(v)})', flush=True)
json.dump(res, open(os.path.join(OUT, 'ssvep_from_raw.json'), 'w'), indent=1)
