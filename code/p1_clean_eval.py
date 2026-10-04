#!/usr/bin/env python3
"""论文一 · 步行/跑步脑电去干扰:几种方法在同一口径下的对照。

数据:Mobile BCI(figshare 版 18 人)。脑电 raw_x 500 Hz(32 导 + 4 眼电),惯性传感器 raw_x 128 Hz(头、左右脚踝各 9 轴),
两者共用同一时钟(事件时间差 < 15 ms,已核)。

统一预处理(所有方法共用):0.5 Hz 高通(5 阶巴特沃斯零相位)→ 32 导平均参考 → 降到 100 Hz。惯性传感器降到 100 Hz,只用加速度+角速度 18 路。
方法(作用在连续数据上,再分段评估):
  none        不处理
  asr / asr10 伪迹子空间重构(meegkit 实现,阈值 20 / 10),用站立训练段校准
  reg         脑电对惯性传感器做带时滞(±100 ms)的岭回归,减去拟合部分
  cca         脑电与带时滞惯性传感器做典型相关,去掉相关系数 > 0.3 的脑电分量(参考通道法,类似 iCanClean 的思路)
  gait        步态同步模板相减:用头部加速度找每一步,把脑电按步相位对齐做局部平均模板(前后各 20 步),逐步减掉(与反搏下心电去干扰同一招)
  gait+reg    先模板相减再回归残余
  reg_head / reg_ankle  消融:回归只用头部 / 只用脚踝传感器
  nlms        在线版:归一化最小均方自适应滤波(逐样本更新,100 ms 延迟即可实时)
  nlms_gated  动/不动门控的在线版:传感器判定无步态的段不处理(站立不受损),有步态才自适应
评估(与原文一致):ERP 用训练段训收缩 LDA、各速度测 AUC;SSVEP 用 CCA 准确率(8 枕区导,0–5 s)。
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_mobile_bci import ERP_CH, ERP_IVALS, SSVEP_CH, SSVEP_DIV, cca_mean_corr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'mobile_bci')
OSF = os.path.join(ROOT, 'data', 'mobile_bci_osf')          # 完整版多出的 sub-19–24(BrainVision, 100 Hz, 73 导含惯性传感器)
OSF_SES = {'tr': '01', '0.0': '02', '0.8': '03', '1.6': '04', '2.0': '05'}
INTERP_BAD = os.environ.get('INTERP_BAD', '1') == '1'            # OSF 版坏道是否插值(默认插)
OUT = os.path.join(ROOT, 'results', 'p1_clean_eval')
FS = 100
SPEEDS = ['0.0', '0.8', '1.6', '2.0']
LAG = 10                                   # ±100 ms
IMU_USE = [i for i in range(27) if (i % 9) < 6]      # 每个传感器的 acc 3 + gyro 3,去掉磁力计


# ---------------- 读数据 ----------------
def load_eeg(path):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    fs = int(d['raw_fs'])
    clab = [str(c) for c in d['raw_clab']][:32]
    x = d['raw_x'].astype(float).T
    eeg = x[:32]
    eeg = sosfiltfilt(butter(5, 0.5, btype='high', fs=fs, output='sos'), eeg, axis=-1)
    eeg = eeg - eeg.mean(0, keepdims=True)
    eeg = resample_poly(eeg, FS, fs, axis=-1)
    on = np.round(np.asarray(d['event'].time, float) / 1000 * FS).astype(int)
    y = np.asarray(d['event'].y)
    return eeg, clab, on, y


def load_imu(path, n):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    fs = int(d['raw_fs'])                                   # 128
    x = d['raw_x'].astype(float).T[IMU_USE]
    x = resample_poly(x, FS * 4, fs, axis=-1)                # 128→400→100,避免 100/128 非整数比
    x = x[:, ::4]
    if x.shape[1] < n:
        x = np.pad(x, ((0, 0), (0, n - x.shape[1])), mode='edge')
    x = x[:, :n]
    x = x - x.mean(1, keepdims=True)
    return x / (x.std(1, keepdims=True) + 1e-9)


def load_osf(sub, task, speed):
    """OSF 版:一个 BrainVision 文件装着 32 头皮 + 14 耳周 + 27 惯性,已是 100 Hz。返回 X, clab, on, y, R 或 None。"""
    import mne
    mne.set_log_level('ERROR')
    f = os.path.join(OSF, sub, f'ses-{OSF_SES[speed]}', 'eeg', f'{sub}_ses-{OSF_SES[speed]}_task-{task}_eeg.vhdr')
    if not os.path.exists(f):
        return None
    raw = mne.io.read_raw_brainvision(f, preload=True)
    names = raw.ch_names
    # 坏道插值(channels.tsv 里 status=bad 的头皮导),与原作者「坏道插值」一步对应;只对 32 头皮导做
    import pandas as pd
    ch = pd.read_csv(f.replace('_eeg.vhdr', '_channels.tsv'), sep='\t')
    bads = [c for c in ch[ch['status'] == 'bad']['name'].tolist() if c in names[:32]]
    if bads and INTERP_BAD:
        scalp = raw.copy().pick(names[:32])
        scalp.set_montage('standard_1020', on_missing='ignore')
        scalp.info['bads'] = bads
        scalp.interpolate_bads(reset_bads=True)
        d = np.concatenate([scalp.get_data(), raw.get_data()[32:]], 0)
    else:
        d = raw.get_data()
    eeg = d[:32] * 1e6
    eeg = sosfiltfilt(butter(5, 0.5, btype='high', fs=FS, output='sos'), eeg, axis=-1)
    eeg = eeg - eeg.mean(0, keepdims=True)
    imu_names = [c for c in names if c.startswith(('Hacc', 'Hgyro', 'Lacc', 'Lgyro', 'Racc', 'Rgyro'))]
    if len(imu_names) < 18:                                   # sub-21 的站立 ERP 段没有惯性通道,整段跳过
        return None
    R = d[[names.index(c) for c in imu_names]]
    R = R - R.mean(1, keepdims=True)
    R = R / (R.std(1, keepdims=True) + 1e-12)
    import pandas as pd
    ev = pd.read_csv(f.replace('_eeg.vhdr', '_events.tsv'), sep='\t')
    on = ev['onset'].to_numpy().astype(int)
    v = ev['value'].to_numpy().astype(int)
    if task == 'ERP':
        y = np.stack([(v == 2).astype(int), (v == 1).astype(int)])        # 2=目标, 1=非目标(原作者 trig_sti)
    else:
        y = np.stack([(v == 11).astype(int), (v == 12).astype(int), (v == 13).astype(int)])   # 5.45 / 8.57 / 12 Hz
    return eeg, names[:32], on, y, R


def get_data(sub, task, speed):
    """统一入口:figshare 版(s01–s18)或 OSF 版(sub-19–24)。"""
    if sub.startswith('sub-'):
        return load_osf(sub, task, speed)
    fe = os.path.join(DATA, f'{sub}_scalp_{task}_{speed}.mat')
    fi = os.path.join(DATA, f'{sub}_IMU_{task}_{speed}.mat')
    if not (os.path.exists(fe) and os.path.exists(fi)):
        return None
    X, clab, on, y = load_eeg(fe)
    return X, clab, on, y, load_imu(fi, X.shape[1])


def lagged(R, lag=LAG):
    """(k, T) → (T, k*(2lag+1)),时滞矩阵。"""
    k, T = R.shape
    cols = []
    for L in range(-lag, lag + 1):
        s = np.roll(R, L, axis=1)
        if L > 0:
            s[:, :L] = 0
        elif L < 0:
            s[:, L:] = 0
        cols.append(s)
    return np.concatenate(cols, 0).T


# ---------------- 方法 ----------------
def m_none(X, R, calib=None):
    return X


def m_asr(X, R, calib=None, cutoff=20):
    """ASR(meegkit 实现)。校准用站立的训练段;若在本段数据上校准,走路伪迹会被当成"正常"而几乎不处理(已实测)。
    平均参考与坏道插值使协方差秩亏,ASR 的重构会发散(OSF 六人上能量比 2–1e18,已实测);
    故先把数据投影到校准协方差的满秩主子空间(特征值 > 1e-8×最大值)再做 ASR,处理后投影回去。"""
    from meegkit.asr import ASR
    C = calib if calib is not None else X
    ev, U = np.linalg.eigh(np.cov(C))
    keep = ev > 1e-8 * ev.max()
    U = U[:, keep]
    asr = ASR(sfreq=FS, cutoff=cutoff, method='euclid')
    asr.fit(U.T @ C)
    return U @ asr.transform(U.T @ X)


def m_asr10(X, R, calib=None):
    return m_asr(X, R, calib, cutoff=10)


def m_reg(X, R, calib=None, alpha=None):
    Z = lagged(R)
    Z = Z - Z.mean(0)
    if alpha is None:
        alpha = 1e-3 * np.trace(Z.T @ Z) / Z.shape[1]
    B = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ X.T)
    return X - (Z @ B).T


def m_cca(X, R, calib=None, thr=0.3, lag=LAG):
    Z = lagged(R, lag)
    Z = Z - Z.mean(0)
    Xc = (X - X.mean(1, keepdims=True)).T                   # (T, 32)
    qx, rx = np.linalg.qr(Xc)
    qz, _ = np.linalg.qr(Z)
    u, s, vt = np.linalg.svd(qx.T @ qz, full_matrices=False)
    k = int((s > thr).sum())
    if k == 0:
        return X
    Wx = np.linalg.solve(rx, u[:, :k])                      # 典型变量 = Xc @ Wx
    V = Xc @ Wx                                             # (T, k)
    B = np.linalg.lstsq(V, Xc, rcond=None)[0]               # 把这些分量从脑电里回归掉
    return X - (V @ B).T


def gait_ratio(R, ch):
    a = np.sqrt((R[ch] ** 2).sum(0))
    a = sosfiltfilt(butter(4, [0.7, 4.0], btype='band', fs=FS, output='sos'), a)
    f = np.fft.rfftfreq(len(a), 1 / FS)
    P = np.abs(np.fft.rfft(a - a.mean())) ** 2
    band = (f > 0.7) & (f < 4)
    return P[band].max() / (P[band].mean() + 1e-12), f[band][np.argmax(P[band])], a


def detect_steps(R):
    """找步点。判据:头部或脚踝加速度模值在 0.7–4 Hz 的频谱峰比 ≥ 25(站立约 9–16,行走 ≥ 28,24 人实测)。
    步点用脚踝加速度找(更可靠)。没有稳定步态返回 None。"""
    rh, fh, _ = gait_ratio(R, [0, 1, 2])
    ra, fa, a = gait_ratio(R, [6, 7, 8])
    if max(rh, ra) < 25:
        return None
    f0 = fa if ra >= rh else fh
    pk, _ = find_peaks(a, distance=int(0.6 / f0 * FS), height=a.std() * 0.5)
    if len(pk) < 30:
        return None
    return pk


def m_gait(X, R, calib=None, nbins=64, half=20):
    pk = detect_steps(R)
    if pk is None:
        return X
    n_cyc = len(pk) - 1
    phase = np.zeros((n_cyc, X.shape[0], nbins))
    for i in range(n_cyc):
        seg = X[:, pk[i]:pk[i + 1]]
        t = np.linspace(0, 1, seg.shape[1])
        tb = np.linspace(0, 1, nbins)
        phase[i] = np.stack([np.interp(tb, t, ch) for ch in seg])
    Y = X.copy()
    cs = np.cumsum(np.concatenate([np.zeros((1,) + phase.shape[1:]), phase], 0), 0)
    for i in range(n_cyc):
        lo, hi = max(0, i - half), min(n_cyc, i + half + 1)
        tpl = (cs[hi] - cs[lo]) / (hi - lo)                   # 前后各 half 步的局部平均模板
        seg_len = pk[i + 1] - pk[i]
        t = np.linspace(0, 1, seg_len)
        tb = np.linspace(0, 1, nbins)
        Y[:, pk[i]:pk[i + 1]] -= np.stack([np.interp(t, tb, ch) for ch in tpl])
    return Y


def m_reg_head(X, R, calib=None):
    """消融:只用头部传感器(前 6 路)。"""
    return m_reg(X, R[:6], calib)


def m_reg_ankle(X, R, calib=None):
    """消融:只用两脚踝传感器(后 12 路)。"""
    return m_reg(X, R[6:], calib)


def m_nlms(X, R, calib=None, mu=0.05, lag=LAG):
    """在线版:归一化最小均方(NLMS)自适应滤波,逐样本更新,只用过去与当前的惯性数据。
    参考含 ±lag 的时滞,实现上把脑电延后 lag 个点(100 ms 延迟)即可实时;这里离线等价计算。"""
    Z = lagged(R, lag)                        # (T, k)
    Z = Z - Z.mean(0)
    T, k = Z.shape
    W = np.zeros((k, X.shape[0]))
    Y = np.empty_like(X)
    eps = 1e-6
    for t in range(T):
        z = Z[t]
        e = X[:, t] - z @ W
        Y[:, t] = e
        W += (mu / (z @ z + eps)) * np.outer(z, e)
    return Y


def m_nlms_gated(X, R, calib=None):
    """动/不动门控的在线自适应滤波:传感器判定整段无步态(站立)就不处理,有步态才自适应。
    (逐样本门控需要传感器的绝对量级,本数据集的惯性数据按段标准化后丢了量级,留作后续。)"""
    if detect_steps(R) is None:
        return X
    return m_nlms(X, R, calib)


def m_nlms_mu02(X, R, calib=None):
    return m_nlms_gated_mu(X, R, 0.02)


def m_nlms_mu10(X, R, calib=None):
    return m_nlms_gated_mu(X, R, 0.10)


def m_nlms_mu20(X, R, calib=None):
    return m_nlms_gated_mu(X, R, 0.20)


def m_nlms_gated_mu(X, R, mu):
    if detect_steps(R) is None:
        return X
    return m_nlms(X, R, None, mu=mu)


def m_cca_thr(X, R, thr):
    return m_cca(X, R, None, thr=thr)


def m_cca20(X, R, calib=None):
    return m_cca_thr(X, R, 0.2)


def m_cca40(X, R, calib=None):
    return m_cca_thr(X, R, 0.4)


def m_reg_shift(X, R, calib=None):
    """对照:惯性参考循环时移 30 s(与脑电失去同步)后再回归。若仍'有效',说明收益不是来自同步伪迹。"""
    return m_reg(X, np.roll(R, 30 * FS, axis=1), calib)


def m_nlms_shift(X, R, calib=None):
    if detect_steps(R) is None:
        return X
    return m_nlms(X, np.roll(R, 30 * FS, axis=1), calib)


def m_reg_a3(X, R, calib=None):
    """岭系数扫描:0.1% → 1%"""
    Z = lagged(R)
    Z = Z - Z.mean(0)
    return m_reg(X, R, calib, alpha=1e-2 * np.trace(Z.T @ Z) / Z.shape[1])


def m_reg_a1(X, R, calib=None):
    """岭系数扫描:0.01%"""
    Z = lagged(R)
    Z = Z - Z.mean(0)
    return m_reg(X, R, calib, alpha=1e-4 * np.trace(Z.T @ Z) / Z.shape[1])


def m_icc(X, R, calib=None, win=2.0, rho2=0.85, lag=0):
    """按 iCanClean 官方实现的口径改编(docs/ref_code/iCanClean/):2 s 滑动统计窗内做脑电与参考的典型相关,
    把 R² > 0.85 的脑电典型分量回归掉;参考 = 18 路惯性(不加时滞;官方默认也无时滞)。与我们的 cca(整段、r>0.3–0.4、±100 ms 时滞)对照。"""
    w = int(win * FS)
    Y = X.copy()
    Z_full = lagged(R, lag) if lag > 0 else R.T
    for t0 in range(0, X.shape[1], w):
        sl = slice(t0, min(t0 + w, X.shape[1]))
        Xw = X[:, sl].T
        Zw = Z_full[sl]
        if Xw.shape[0] < 2 * Xw.shape[1]:
            continue
        Xc = Xw - Xw.mean(0)
        Zc = Zw - Zw.mean(0)
        qx, rx = np.linalg.qr(Xc)
        qz, _ = np.linalg.qr(Zc)
        u, sv, vt = np.linalg.svd(qx.T @ qz, full_matrices=False)
        k = int((sv ** 2 > rho2).sum())
        if k == 0:
            continue
        Wx = np.linalg.solve(rx, u[:, :k])
        V = Xc @ Wx
        B = np.linalg.lstsq(V, Xc, rcond=None)[0]
        Y[:, sl] = (Xw - V @ B).T
    return Y


def m_icc50(X, R, calib=None):
    return m_icc(X, R, calib, rho2=0.5)


def m_icc_w4(X, R, calib=None):
    return m_icc(X, R, calib, win=4.0)


def _shift(R, sec):
    return np.roll(R, int(sec * FS), axis=1)


def _surrogate(R, seed=0):
    """相位随机化替代:保留每路惯性信号的功率谱(含步频周期性),打乱相位,与脑电失去任何时间对应。"""
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(R, axis=1)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape))
    ph[:, 0] = 1
    return np.fft.irfft(np.abs(F) * ph, n=R.shape[1], axis=1)


def m_reg_shift10(X, R, calib=None):
    return m_reg(X, _shift(R, 10), calib)


def m_reg_shift120(X, R, calib=None):
    return m_reg(X, _shift(R, 120), calib)


def m_reg_surr(X, R, calib=None):
    return m_reg(X, _surrogate(R), calib)


def m_nlms_shift10(X, R, calib=None):
    return X if detect_steps(R) is None else m_nlms(X, _shift(R, 10), calib)


def m_nlms_shift120(X, R, calib=None):
    return X if detect_steps(R) is None else m_nlms(X, _shift(R, 120), calib)


def m_nlms_surr(X, R, calib=None):
    return X if detect_steps(R) is None else m_nlms(X, _surrogate(R), calib)


def m_gait_reg(X, R, calib=None):
    return m_reg(m_gait(X, R), R)


METHODS = {'none': m_none, 'asr': m_asr, 'asr10': m_asr10, 'reg': m_reg, 'cca': m_cca, 'gait': m_gait, 'gait+reg': m_gait_reg,
           'reg_head': m_reg_head, 'reg_ankle': m_reg_ankle, 'nlms': m_nlms, 'nlms_gated': m_nlms_gated,
           'nlms_mu02': m_nlms_mu02, 'nlms_mu10': m_nlms_mu10, 'nlms_mu20': m_nlms_mu20, 'cca20': m_cca20, 'cca40': m_cca40,
           'reg_shift': m_reg_shift, 'nlms_shift': m_nlms_shift, 'reg_a1': m_reg_a1, 'reg_a3': m_reg_a3,
           'icc': m_icc, 'icc50': m_icc50, 'icc_w4': m_icc_w4,
           'reg_shift10': m_reg_shift10, 'reg_shift120': m_reg_shift120, 'reg_surr': m_reg_surr,
           'nlms_shift10': m_nlms_shift10, 'nlms_shift120': m_nlms_shift120, 'nlms_surr': m_nlms_surr}


# ---------------- 审稿补算(v4):分块回归、CCA 2×2、r 网格延长 ----------------
def m_reg_block(X, R, calib=None, block_s=30):
    """分块岭回归:每 block_s 秒独立拟合(局部平稳假设),用于区分 NLMS 的优势来自"自适应"还是"算法"。
    末块不足半块时并入前一块;块内时滞矩阵边界补零(与整段版相同的处理)。"""
    T = X.shape[1]
    B = int(block_s * FS)
    edges = list(range(0, T, B))
    if len(edges) > 1 and T - edges[-1] < B // 2:
        edges = edges[:-1]
    edges.append(T)
    Y = X.copy()
    for a, b in zip(edges[:-1], edges[1:]):
        Y[:, a:b] = m_reg(X[:, a:b], R[:, a:b], calib)
    return Y


def m_reg_b10(X, R, calib=None):
    return m_reg_block(X, R, calib, block_s=10)


def m_reg_b30(X, R, calib=None):
    return m_reg_block(X, R, calib, block_s=30)


def m_cca40_nolag(X, R, calib=None):
    """CCA 2×2 的一格:整段、无时滞、r>0.4。"""
    return m_cca(X, R, calib, thr=0.4, lag=0)


def m_cca50(X, R, calib=None):
    return m_cca_thr(X, R, 0.5)


def m_cca60(X, R, calib=None):
    return m_cca_thr(X, R, 0.6)


def m_cca70(X, R, calib=None):
    return m_cca_thr(X, R, 0.7)


def m_icc_w4_lag(X, R, calib=None):
    """CCA 2×2 的一格:4 s 窗 + ±100 ms 时滞(378 个回归量对 400 个采样点,预期退化,如实报)。"""
    return m_icc(X, R, calib, win=4.0, lag=LAG)


def m_icc_w20(X, R, calib=None):
    return m_icc(X, R, calib, win=20.0)


def m_icc_w20_lag(X, R, calib=None):
    return m_icc(X, R, calib, win=20.0, lag=LAG)


METHODS.update({'reg_b10': m_reg_b10, 'reg_b30': m_reg_b30, 'cca40_nolag': m_cca40_nolag,
                'cca50': m_cca50, 'cca60': m_cca60, 'cca70': m_cca70,
                'icc_w4_lag': m_icc_w4_lag, 'icc_w20': m_icc_w20, 'icc_w20_lag': m_icc_w20_lag})


# ---------------- 评估 ----------------
def erp_feats(X, clab, on, y):
    idx = [clab.index(c) for c in ERP_CH if c in clab]
    t = np.arange(-20, 80) * 10                              # -200 … 790 ms
    ep = np.stack([X[idx][:, o - 20:o + 80] for o in on if o - 20 >= 0 and o + 80 <= X.shape[1]], -1)   # (ch, 100, 试次)
    yy = y[0][:ep.shape[-1]]
    ep = ep - ep[:, (t >= -200) & (t <= 0), :].mean(1, keepdims=True)
    f = np.stack([ep[:, (t > a) & (t <= b), :].mean(1) for a, b in ERP_IVALS], 0)   # (5, ch, 试次)
    return f.reshape(-1, f.shape[-1]).T, yy


def ssvep_acc(X, clab, on, y):
    idx = [clab.index(c) for c in SSVEP_CH]
    n = 5 * FS
    tt = np.arange(1, n + 1) / FS
    refs = [np.stack([np.sin(2 * np.pi * 60 / k * tt), np.cos(2 * np.pi * 60 / k * tt),
                      np.sin(4 * np.pi * 60 / k * tt), np.cos(4 * np.pi * 60 / k * tt)], 1) for k in SSVEP_DIV]
    pred, lab = [], []
    for i, o in enumerate(on):
        if o + n > X.shape[1]:
            continue
        pred.append(int(np.argmax([cca_mean_corr(X[idx, o:o + n].T, r) for r in refs])))
        lab.append(int(np.argmax(y[:, i])))
    return float(np.mean(np.array(pred) == np.array(lab)) * 100)


def run_subject(sub, methods):
    res = {}
    calib = None
    for sp in ('tr', '0.0'):
        for task in ('ERP', 'SSVEP'):
            g = get_data(sub, task, sp)
            if g is not None:
                calib = g[0]
                break
        if calib is not None:
            break
    clf0 = None
    g = get_data(sub, 'ERP', 'tr')
    if g is not None:
        X, clab, on, y, _ = g
        f, yy = erp_feats(X, clab, on, y)
        clf0 = LDA(solver='lsqr', shrinkage='auto').fit(f, yy)
    for sp in SPEEDS:
        for task in ('ERP', 'SSVEP'):
            g = get_data(sub, task, sp)
            if g is None:
                continue
            X, clab, on, y, R = g
            for m in methods:
                try:
                    Xc = METHODS[m](X, R, calib)
                    if task == 'ERP':
                        if clf0 is None:
                            continue
                        f, yy = erp_feats(Xc, clab, on, y)
                        res[f'ERP_{sp}_{m}'] = float(roc_auc_score(yy, clf0.decision_function(f)))
                    else:
                        res[f'SSVEP_{sp}_{m}'] = ssvep_acc(Xc, clab, on, y)
                except Exception as e:
                    print(sub, task, sp, m, 'ERR', e, flush=True)
    print(sub, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
    return sub, res


def main():
    os.makedirs(OUT, exist_ok=True)
    methods = sys.argv[1].split(',') if len(sys.argv) > 1 else list(METHODS)
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    subs = sorted({re.match(r'(s\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(DATA, 's*_scalp_*.mat'))})
    subs += sorted(os.path.basename(d) for d in glob.glob(os.path.join(OSF, 'sub-*')) if os.path.isdir(d))
    if os.environ.get('SUBS'):
        subs = [x for x in subs if x in os.environ['SUBS'].split(',')]
    rows = dict(Parallel(n_jobs=n_jobs)(delayed(run_subject)(s, methods) for s in subs))
    summ = {}
    for task in ('ERP', 'SSVEP'):
        for sp in SPEEDS:
            for m in methods:
                v = [r[f'{task}_{sp}_{m}'] for r in rows.values() if f'{task}_{sp}_{m}' in r]
                if v:
                    summ[f'{task}_{sp}_{m}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0, 'n': len(v)}
    tag = sys.argv[3] if len(sys.argv) > 3 else 'results'
    json.dump({'per_subject': rows, 'summary': summ}, open(os.path.join(OUT, f'{tag}.json'), 'w'), indent=1)
    print('\n=== 汇总(均值, 人数) ===')
    for task in ('ERP', 'SSVEP'):
        print(task, '      ' + '  '.join(f'{m:>9s}' for m in methods))
        for sp in SPEEDS:
            cells = []
            for m in methods:
                s = summ.get(f'{task}_{sp}_{m}')
                cells.append(f'{s["mean"]:9.3f}' if (s and task == 'ERP') else (f'{s["mean"]:9.1f}' if s else ' ' * 9))
            n = summ.get(f'{task}_{sp}_{methods[0]}', {}).get('n', 0)
            print(f'  {sp} m/s ' + '  '.join(cells) + f'   (n={n})')


if __name__ == '__main__':
    main()
