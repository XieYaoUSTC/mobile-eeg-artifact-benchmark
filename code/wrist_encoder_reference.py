#!/usr/bin/env python3
"""腕部外骨骼数据:用「器械自身的编码器角度」作参考去干扰,看运动期脑电指标是否更干净。

这是「器械自身信号作参考」在公开数据上唯一的直接检验。注意:外骨骼在透明模式(电机断电),人主动动腕,
伪迹来源主要是手臂/头颈动作带来的电极微动与肌电,量级可能不大;结果不好也要如实报。

做法(每人、每个任务段):
  参考信号 = 编码器 X、Y 角度(100 Hz 更新,与 512 Hz 脑电同文件同步)及其一阶差分(角速度),带 ±100 ms 时滞做岭回归,
  从 8 导脑电里减去拟合部分;对照 = 不处理。两种口径下各算:
  (a) 运动期 vs 预览期的 mu/beta 功率下降(ERD%)在 C3 —— 期望去干扰后更负、跨试次更稳定(变异系数更小)
  (b) 运动期 1–40 Hz 宽带功率相对预览期的比值 —— 伪迹若被去掉,这个比值应下降
  (c) 单试次 ERD 的跨试次标准差 —— 衡量稳定性
  (d) 用运动期 vs 预览期做「动没动」的检测 AUC(C3+C4 功率,留一法 LDA)—— 下游任务口径
"""
import glob
import json
import os
import warnings

import mne
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'wrist_exo')
OUT = os.path.join(ROOT, 'results', 'wrist_encoder_reference')
EEG = ['C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CPz', 'CP4']
FS = 512
LAG = int(0.1 * FS)      # ±100 ms
TRIAL, REF, MOV = 10.0, (3.0, 5.0), (5.0, 7.5)


def lagged(R, lag, step=4):
    """时滞矩阵;每 step 个采样取一个时滞,控制回归量个数(512 Hz 下 ±100 ms 共 ±51 点,取 26 个时滞)。"""
    cols = []
    for L in range(-lag, lag + 1, step):
        s = np.roll(R, L, axis=1)
        if L > 0:
            s[:, :L] = 0
        elif L < 0:
            s[:, L:] = 0
        cols.append(s)
    return np.concatenate(cols, 0).T


def reg_out(X, R):
    Z = lagged(R, LAG)
    Z = Z - Z.mean(0)
    alpha = 1e-3 * np.trace(Z.T @ Z) / Z.shape[1]
    B = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ X.T)
    return X - (Z @ B).T


def band_power(x, lo, hi):
    sos = butter(4, [lo, hi], btype='band', fs=FS, output='sos')
    return np.abs(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)) ** 2


def metrics(eeg, n_trials):
    n = int(TRIAL * FS)
    r0, r1, m0, m1 = (int(t * FS) for t in (*REF, *MOV))
    pw = {'mu': band_power(eeg, 8, 13), 'beta': band_power(eeg, 13, 30), 'broad': band_power(eeg, 1, 40)}
    c3, c4 = EEG.index('C3'), EEG.index('C4')
    erd_mu, erd_beta, broad_ratio, feats = [], [], [], []
    for i in range(n_trials):
        s = i * n
        f = []
        for b in ('mu', 'beta'):
            pr = pw[b][:, s + r0:s + r1].mean(-1)
            pm = pw[b][:, s + m0:s + m1].mean(-1)
            if b == 'mu':
                erd_mu.append((pm[c3] - pr[c3]) / pr[c3] * 100)
            else:
                erd_beta.append((pm[c3] - pr[c3]) / pr[c3] * 100)
            f += [np.log(pr[c3]), np.log(pr[c4]), np.log(pm[c3]), np.log(pm[c4])]
        pr = pw['broad'][:, s + r0:s + r1].mean()
        pm = pw['broad'][:, s + m0:s + m1].mean()
        broad_ratio.append(pm / pr)
        feats.append(f)
    feats = np.array(feats)
    # 「动没动」检测:每试次给出(参考期特征)和(运动期特征)两条样本
    Xa = np.c_[feats[:, [0, 1, 4, 5]]]          # 参考期 mu/beta C3 C4
    Xb = np.c_[feats[:, [2, 3, 6, 7]]]          # 运动期
    Xd = np.r_[Xa, Xb]
    yd = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))].astype(int)
    grp = np.r_[np.arange(len(Xa)), np.arange(len(Xb))]
    pred = np.zeros(len(yd))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, grp % 5):
        pred[te] = LDA().fit(Xd[tr], yd[tr]).decision_function(Xd[te])
    return {'erd_mu_C3': float(np.median(erd_mu)), 'erd_beta_C3': float(np.median(erd_beta)),
            'erd_mu_sd': float(np.std(erd_mu)), 'broad_ratio': float(np.median(broad_ratio)),
            'move_auc': float(roc_auc_score(yd, pred))}


def run_file(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    d = raw.get_data()
    eeg = d[:8] * (1e6 if np.nanmax(np.abs(d[:8])) < 1 else 1.0)
    eeg = eeg - eeg.mean(0, keepdims=True)
    eeg = sosfiltfilt(butter(4, [0.5, 45], btype='band', fs=FS, output='sos'), eeg, axis=-1)
    mfile = path.replace('/eeg/', '/motion/').replace('_task-WristPointingTask_run', '_task-WristPointingTask_tracksys-BiomechWrist_run').replace('_eeg.edf', '_motion.tsv')
    xy = np.loadtxt(mfile, delimiter='\t').T[:, :d.shape[1]].astype(float)
    if max(len(np.unique(xy[0])), len(np.unique(xy[1]))) < 50:
        return None
    xy = np.where(xy < 1000, np.nan, xy)
    xy = np.where(np.isnan(xy), np.nanmedian(xy, axis=1, keepdims=True), xy)
    vel = np.gradient(xy, axis=1)
    R = np.r_[xy, vel]
    R = R - R.mean(1, keepdims=True)
    R = R / (R.std(1, keepdims=True) + 1e-9)
    n_trials = int(d.shape[1] // int(TRIAL * FS))
    out = {'none': metrics(eeg, n_trials), 'enc_reg': metrics(reg_out(eeg, R), n_trials)}
    out['var_removed_pct'] = float((1 - np.var(reg_out(eeg, R)) / np.var(eeg)) * 100)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in sorted(d for d in glob.glob(os.path.join(DATA, 'sub-*')) if os.path.isdir(d)):
        name = os.path.basename(sub)
        acc = {'none': [], 'enc_reg': [], 'var': []}
        for f in sorted(glob.glob(os.path.join(sub, 'eeg', '*_eeg.edf'))):
            try:
                o = run_file(f)
            except Exception as e:
                print(name, os.path.basename(f), 'ERR', e, flush=True)
                continue
            if o is None:
                continue
            acc['none'].append(o['none'])
            acc['enc_reg'].append(o['enc_reg'])
            acc['var'].append(o['var_removed_pct'])
        if not acc['none']:
            continue
        r = {'sub': name, 'var_removed_pct': float(np.mean(acc['var']))}
        for cond in ('none', 'enc_reg'):
            for k in acc[cond][0]:
                r[f'{cond}_{k}'] = float(np.mean([m[k] for m in acc[cond]]))
        rows.append(r)
        print(name, f"去掉方差 {r['var_removed_pct']:.1f}%  ERD_mu_C3 {r['none_erd_mu_C3']:.1f}→{r['enc_reg_erd_mu_C3']:.1f}  试次间SD {r['none_erd_mu_sd']:.1f}→{r['enc_reg_erd_mu_sd']:.1f}  宽带比 {r['none_broad_ratio']:.2f}→{r['enc_reg_broad_ratio']:.2f}  动没动AUC {r['none_move_auc']:.3f}→{r['enc_reg_move_auc']:.3f}", flush=True)
    keys = [k for k in rows[0] if k != 'sub']
    summ = {k: {'mean': float(np.mean([r[k] for r in rows])), 'sd': float(np.std([r[k] for r in rows], ddof=1))} for k in keys}
    from scipy.stats import wilcoxon
    for k in ('erd_mu_C3', 'erd_beta_C3', 'erd_mu_sd', 'broad_ratio', 'move_auc'):
        a = np.array([r[f'none_{k}'] for r in rows])
        b = np.array([r[f'enc_reg_{k}'] for r in rows])
        summ[f'paired_{k}'] = {'mean_diff': float((b - a).mean()), 'n_up': int((b > a).sum()), 'n_down': int((b < a).sum()),
                               'p': float(wilcoxon(b, a).pvalue) if np.any(b != a) else 1.0}
    json.dump({'rows': rows, 'summary': summ}, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 汇总(%d 人) ===' % len(rows))
    for k in ('erd_mu_C3', 'erd_beta_C3', 'erd_mu_sd', 'broad_ratio', 'move_auc'):
        print(f"{k:12s} 不处理 {summ['none_'+k]['mean']:.3f}  编码器回归 {summ['enc_reg_'+k]['mean']:.3f}  差 {summ['paired_'+k]['mean_diff']:+.3f}  升/降 {summ['paired_'+k]['n_up']}/{summ['paired_'+k]['n_down']}  p={summ['paired_'+k]['p']:.3f}")
    print(f"去掉的方差占比 {summ['var_removed_pct']['mean']:.1f}%")


if __name__ == '__main__':
    main()
