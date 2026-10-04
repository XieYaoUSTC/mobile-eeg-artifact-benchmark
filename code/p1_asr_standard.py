#!/usr/bin/env python3
"""论文一 · 审稿补算:按标准顺序跑 ASR(ASR 在重参考与坏道插值之前)。

标准顺序(clean_rawdata 的惯例):原始参考数据 → 0.5 Hz 高通 → 降到 100 Hz → 去掉坏道 → ASR(站立训练段校准)→ 插值坏道 → 平均参考。
与 p1_clean_eval.m_asr 的区别:那条线先 CAR + 插值再 ASR,协方差秩亏,需要"rank-safe"投影;本脚本不需要任何投影。
figshare 版:无坏道标注,32 导满秩;OSF 版:channels.tsv 标注的坏道在 ASR 前剔除,ASR 后用 MNE 球面插值。
输出 results/p1_clean_eval/results_24_asr_std.json:per_subject 含 ERP/SSVEP_{speed}_{none,asr_std,asr10_std},
energy_ratio 记录每段 ASR 输出/输入能量比(验证无发散)。
用法: p1_asr_standard.py [并行数]
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_clean_eval as p  # noqa: E402

FS = p.FS
OUT = p.OUT


def load_eeg_nocar(path):
    """figshare:高通 + 降采样,不做 CAR。返回 eeg(32,T), clab, on, y, bads=[]。"""
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    fs = int(d['raw_fs'])
    clab = [str(c) for c in d['raw_clab']][:32]
    x = d['raw_x'].astype(float).T[:32]
    x = sosfiltfilt(butter(5, 0.5, btype='high', fs=fs, output='sos'), x, axis=-1)
    x = resample_poly(x, FS, fs, axis=-1)
    on = np.round(np.asarray(d['event'].time, float) / 1000 * FS).astype(int)
    y = np.asarray(d['event'].y)
    return x, clab, on, y, []


def load_osf_nocar(sub, task, speed):
    """OSF:高通,不插值、不 CAR;返回 eeg, clab, on, y, bads, R。"""
    import mne
    mne.set_log_level('ERROR')
    f = os.path.join(p.OSF, sub, f'ses-{p.OSF_SES[speed]}', 'eeg', f'{sub}_ses-{p.OSF_SES[speed]}_task-{task}_eeg.vhdr')
    if not os.path.exists(f):
        return None
    raw = mne.io.read_raw_brainvision(f, preload=True)
    names = raw.ch_names
    ch = pd.read_csv(f.replace('_eeg.vhdr', '_channels.tsv'), sep='\t')
    bads = [c for c in ch[ch['status'] == 'bad']['name'].tolist() if c in names[:32]]
    d = raw.get_data()
    eeg = d[:32] * 1e6
    eeg = sosfiltfilt(butter(5, 0.5, btype='high', fs=FS, output='sos'), eeg, axis=-1)
    imu_names = [c for c in names if c.startswith(('Hacc', 'Hgyro', 'Lacc', 'Lgyro', 'Racc', 'Rgyro'))]
    if len(imu_names) < 18:
        return None
    R = d[[names.index(c) for c in imu_names]]
    R = R - R.mean(1, keepdims=True)
    R = R / (R.std(1, keepdims=True) + 1e-12)
    ev = pd.read_csv(f.replace('_eeg.vhdr', '_events.tsv'), sep='\t')
    on = ev['onset'].to_numpy().astype(int)
    v = ev['value'].to_numpy().astype(int)
    if task == 'ERP':
        y = np.stack([(v == 2).astype(int), (v == 1).astype(int)])
    else:
        y = np.stack([(v == 11).astype(int), (v == 12).astype(int), (v == 13).astype(int)])
    return eeg, names[:32], on, y, bads, R


def get_nocar(sub, task, speed):
    if sub.startswith('sub-'):
        return load_osf_nocar(sub, task, speed)
    fe = os.path.join(p.DATA, f'{sub}_scalp_{task}_{speed}.mat')
    fi = os.path.join(p.DATA, f'{sub}_IMU_{task}_{speed}.mat')
    if not (os.path.exists(fe) and os.path.exists(fi)):
        return None
    X, clab, on, y, bads = load_eeg_nocar(fe)
    return X, clab, on, y, bads, p.load_imu(fi, X.shape[1])


def interp_and_car(X, clab, bads):
    """坏道球面插值(MNE)后平均参考。"""
    if bads:
        import mne
        mne.set_log_level('ERROR')
        info = mne.create_info(clab, FS, 'eeg')
        raw = mne.io.RawArray(X * 1e-6, info)
        raw.set_montage('standard_1020', on_missing='ignore')
        raw.info['bads'] = list(bads)
        raw.interpolate_bads(reset_bads=True)
        X = raw.get_data() * 1e6
    return X - X.mean(0, keepdims=True)


def asr_standard(X, calib, good, cutoff):
    """ASR 只在好导上做(校准段与测试段共同的好导),返回处理后的 X(坏导原样)与能量比。"""
    from meegkit.asr import ASR
    asr = ASR(sfreq=FS, cutoff=cutoff, method='euclid')
    asr.fit(calib[good])
    Y = X.copy()
    Y[good] = asr.transform(X[good])
    ratio = float((Y[good] ** 2).sum() / ((X[good] ** 2).sum() + 1e-12))
    return Y, ratio


def run_subject(sub):
    res, ratios = {}, {}
    calib = None
    for sp in ('tr', '0.0'):
        for task in ('ERP', 'SSVEP'):
            g = get_nocar(sub, task, sp)
            if g is not None:
                calib, calib_bads = g[0], g[4]
                break
        if calib is not None:
            break
    if calib is None:
        return sub, {}, {}
    clf0 = None
    g = get_nocar(sub, 'ERP', 'tr')
    if g is not None:
        X, clab, on, y, bads, _ = g
        f, yy = p.erp_feats(interp_and_car(X, clab, bads), clab, on, y)
        clf0 = LDA(solver='lsqr', shrinkage='auto').fit(f, yy)
    for sp in p.SPEEDS:
        for task in ('ERP', 'SSVEP'):
            g = get_nocar(sub, task, sp)
            if g is None:
                continue
            X, clab, on, y, bads, R = g
            allbad = sorted(set(bads) | set(calib_bads))
            good = [i for i, c in enumerate(clab) if c not in allbad]
            for name, cutoff in (('none', None), ('asr_std', 20), ('asr10_std', 10)):
                try:
                    if cutoff is None:
                        Xc = X
                    else:
                        Xc, ratios[f'{task}_{sp}_{name}'] = asr_standard(X, calib, good, cutoff)
                    Xc = interp_and_car(Xc, clab, allbad)
                    if task == 'ERP':
                        if clf0 is None:
                            continue
                        f, yy = p.erp_feats(Xc, clab, on, y)
                        res[f'{task}_{sp}_{name}'] = float(roc_auc_score(yy, clf0.decision_function(f)))
                    else:
                        res[f'{task}_{sp}_{name}'] = p.ssvep_acc(Xc, clab, on, y)
                except Exception as e:
                    print(sub, task, sp, name, 'ERR', e, flush=True)
    print(sub, json.dumps({k: round(v, 3) for k, v in res.items()}), 'ratio', json.dumps({k: round(v, 3) for k, v in ratios.items()}), flush=True)
    return sub, res, ratios


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    subs = sorted({re.match(r'(s\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(p.DATA, 's*_scalp_*.mat'))})
    subs += sorted(os.path.basename(d) for d in glob.glob(os.path.join(p.OSF, 'sub-*')) if os.path.isdir(d))
    if os.environ.get('SUBS'):
        subs = [x for x in subs if x in os.environ['SUBS'].split(',')]
    out = Parallel(n_jobs=n_jobs)(delayed(run_subject)(s) for s in subs)
    rows = {s: r for s, r, _ in out}
    ratios = {s: r for s, _, r in out}
    summ = {}
    for task in ('ERP', 'SSVEP'):
        for sp in p.SPEEDS:
            for m in ('none', 'asr_std', 'asr10_std'):
                v = [r[f'{task}_{sp}_{m}'] for r in rows.values() if f'{task}_{sp}_{m}' in r]
                if v:
                    summ[f'{task}_{sp}_{m}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0, 'n': len(v)}
    json.dump({'per_subject': rows, 'summary': summ, 'energy_ratio': ratios,
               'note': 'ASR before interpolation and CAR (standard order); calibration = standing training session, good channels common to calibration and test'},
              open(os.path.join(OUT, 'results_24_asr_std' + os.environ.get('OUT_TAG', '') + '.json'), 'w'), indent=1)
    print('\n=== 标准顺序 ASR 汇总 ===')
    for task in ('ERP', 'SSVEP'):
        for sp in p.SPEEDS:
            cells = [summ.get(f'{task}_{sp}_{m}', {}).get('mean', float('nan')) for m in ('none', 'asr_std', 'asr10_std')]
            print(f'  {task} {sp} ' + '  '.join(f'{c:.3f}' if task == 'ERP' else f'{c:.1f}' for c in cells) + f"  (n={summ.get(f'{task}_{sp}_none', {}).get('n', 0)})")
    allr = [v for r in ratios.values() for v in r.values()]
    print('能量比 min/median/max:', min(allr), float(np.median(allr)), max(allr))


if __name__ == '__main__':
    main()
