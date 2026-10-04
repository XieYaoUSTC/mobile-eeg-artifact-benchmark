#!/usr/bin/env python3
"""Mobile BCI 数据集(Lee et al., Sci Data 2021)头皮脑电基线复现。

对齐原作者 MATLAB 代码(github youngeun1209/MobileBCI_Data)的口径,直接用数据集里
已预处理并分段的 preprocess_x(100 Hz):
  ERP   训练段(tr)上训练收缩 LDA,在各速度段上测试,指标 AUC
        通道 17 个;基线 -200~0 ms;特征=200~450 ms 内每 50 ms 的均值(5 段)
  SSVEP 无训练的典型相关分析(CCA),参考信号=基频与二倍频的正余弦
        通道 8 个枕区;刺激 60/11、60/7、60/5 Hz;取典型相关系数均值最大者
原文头皮脑电结果(0 / 0.8 / 1.6 / 2.0 m/s):
  ERP AUC   0.90 / 0.77 / 0.67 / 0.58
  SSVEP 准确率 88.70 / 83.12 / 80.65 / 54.76 %
"""
import glob
import json
import os
import re

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'mobile_bci')
OUT = os.path.join(ROOT, 'results', 'mobile_bci_baseline')
SPEEDS = ['0.0', '0.8', '1.6', '2.0']
ERP_CH = ['C3', 'C1', 'C2', 'C4', 'CP1', 'CP2', 'P3', 'Pz', 'P4', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
SSVEP_CH = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
ERP_IVALS = [(200, 250), (250, 300), (300, 350), (350, 400), (400, 450)]
_meta = os.path.join(DATA, '_figshare_meta.json')
SIZES = {f['name']: f['size'] for f in json.load(open(_meta))['files']} if os.path.exists(_meta) else {}
SSVEP_DIV = [11, 7, 5]          # 60/11=5.45, 60/7=8.57, 60/5=12 Hz;与 event.className 的顺序一致


def complete(path):
    """文件存在且大小与 figshare 清单一致(下载未完成的文件跳过)。"""
    if not os.path.exists(path):
        return False
    want = SIZES.get(os.path.basename(path))
    return want is None or os.path.getsize(path) == want


def load(path):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    clab = [str(c) for c in d['preprocess_clab']]
    return d['preprocess_x'].astype(float), clab, np.asarray(d['t'], float), d['event'].y, int(d['preprocess_fs'])


def pick(x, clab, want):
    idx = [clab.index(c) for c in want if c in clab]      # 数据里没有的通道(如 C1/C2)自动跳过
    return x[:, idx, :], len(idx)


def erp_features(x, clab, t):
    x, n_ch = pick(x, clab, ERP_CH)                        # (时间, 通道, 试次)
    x = x - x[(t >= -200) & (t <= 0)].mean(0, keepdims=True)
    f = np.stack([x[(t > a) & (t <= b)].mean(0) for a, b in ERP_IVALS], 0)   # (5, 通道, 试次)
    return f.reshape(-1, f.shape[-1]).T, n_ch


def cca_mean_corr(X, Y):
    """X: (样本, p), Y: (样本, q)。返回各典型相关系数的均值。"""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def ssvep_predict(x, clab, fs):
    x, n_ch = pick(x, clab, SSVEP_CH)
    n = x.shape[0]
    tt = np.arange(1, n + 1) / fs
    refs = [np.stack([np.sin(2 * np.pi * 60 / d * tt), np.cos(2 * np.pi * 60 / d * tt),
                      np.sin(2 * np.pi * 2 * 60 / d * tt), np.cos(2 * np.pi * 2 * 60 / d * tt)], 1) for d in SSVEP_DIV]
    pred = np.array([int(np.argmax([cca_mean_corr(x[:, :, i], r) for r in refs])) for i in range(x.shape[2])])
    return pred, n_ch


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = sorted({re.match(r'(s\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(DATA, 's*_scalp_*.mat'))})
    erp, ssvep = {s: {} for s in SPEEDS}, {s: {} for s in SPEEDS}
    for sub in subs:
        ftr = os.path.join(DATA, f'{sub}_scalp_ERP_tr.mat')
        clf = None
        if complete(ftr):
            x, clab, t, y, _ = load(ftr)
            ftr_x, _ = erp_features(x, clab, t)
            clf = LDA(solver='lsqr', shrinkage='auto').fit(ftr_x, y[0])      # y[0]=1 表示目标刺激
        for sp in SPEEDS:
            f = os.path.join(DATA, f'{sub}_scalp_ERP_{sp}.mat')
            if clf is not None and complete(f):
                x, clab, t, y, _ = load(f)
                fx, _ = erp_features(x, clab, t)
                erp[sp][sub] = float(roc_auc_score(y[0], clf.decision_function(fx)))
            f = os.path.join(DATA, f'{sub}_scalp_SSVEP_{sp}.mat')
            if complete(f):
                x, clab, t, y, fs = load(f)
                pred, _ = ssvep_predict(x, clab, fs)
                ssvep[sp][sub] = float(np.mean(pred == np.argmax(y, 0)) * 100)
        print(sub, 'ERP', {s: round(erp[s][sub], 2) for s in SPEEDS if sub in erp[s]},
              'SSVEP', {s: round(ssvep[s][sub], 1) for s in SPEEDS if sub in ssvep[s]}, flush=True)
    summ = {}
    print('\n=== 汇总(均值 ± 标准差, 人数) ===   原文 ERP 0.90/0.77/0.67/0.58   SSVEP 88.70/83.12/80.65/54.76')
    for name, tab in (('ERP_AUC', erp), ('SSVEP_ACC', ssvep)):
        for sp in SPEEDS:
            v = list(tab[sp].values())
            if v:
                summ[f'{name}_{sp}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'n': len(v)}
                print(f'{name} {sp} m/s: {np.mean(v):.2f} ± {summ[f"{name}_{sp}"]["sd"]:.2f}  (n={len(v)})')
    json.dump({'erp': erp, 'ssvep': ssvep, 'summary': summ}, open(os.path.join(OUT, 'baseline.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
