#!/usr/bin/env python3
"""论文一 · 因果回放:整条流水线只用过去的数据,逐样本前进。

与 p1_clean_eval.py(回顾性、零相位)的区别:
  前端   因果 2 阶巴特沃斯 0.5 Hz 高通;平均参考;500→100 Hz 用因果 4 阶 40 Hz 低通 + 抽样(OSF 版已是 100 Hz 只做高通)
  惯性   128→100 Hz 零阶保持(严格因果,延迟 ≤ 7.8 ms);滑动归一化(指数遗忘,时间常数 10 s,前 5 s 初始化)
  门控   头部加速度 0.7–4 Hz 的尾随 2 s 均方,相对本人站立校准段(训练段)中位数的倍数;每 1 s 判一次,连续 3 次 ≥ 阈值开门、连续 3 次 < 0.4×阈值关门(滞回);判定作用于下一秒
         (尾随短窗的频谱峰比在站立/行走之间重叠,不可用,已实测)
  NLMS   把脑电延后 L=10 个点(100 ms),回归量用惯性 t−2L…t 共 21 个时滞;输出对应 t−L 时刻的脑电
  延迟   100 ms(时滞缓冲)+ 因果滤波群延迟(高通 <10 ms,低通约 10 ms)+ 门控决策粒度 1 s
评估与回顾性版本相同(ERP 站立训练段训收缩 LDA 测 AUC;SSVEP CCA 准确率),分类器/参考不变。
另做「过渡」实验:同一人同一任务,把站立段与行走段首尾相接,看门控多久切换、站立部分是否被改动。
用法: p1_causal_replay.py [并行数] [门控阈值,默认 25] [步长,默认 0.05]
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
from scipy.signal import butter, lfilter, find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_clean_eval as p  # noqa: E402

FS = 100
OUT = os.path.join(p.ROOT, 'results', 'p1_causal_replay')
LAG = 10
HP = float(os.environ.get('CAUSAL_HP', '0.5'))     # 因果高通截止(Hz);审稿补算用 0.1 做前端敏感性


# ---------------- 因果前端 ----------------
def causal_eeg(path_or_tuple):
    """返回 100 Hz 的因果预处理脑电 (32, T)、clab、on、y。"""
    sub, task, sp = path_or_tuple
    if sub.startswith('sub-'):
        g = p.load_osf(sub, task, sp)                      # 已是 100 Hz;其内部做的是零相位高通,这里重做因果高通
        if g is None:
            return None
        import mne
        mne.set_log_level('ERROR')
        f = os.path.join(p.OSF, sub, f'ses-{p.OSF_SES[sp]}', 'eeg', f'{sub}_ses-{p.OSF_SES[sp]}_task-{task}_eeg.vhdr')
        raw = mne.io.read_raw_brainvision(f, preload=True)
        names = raw.ch_names
        d = raw.get_data()
        eeg = d[:32] * 1e6
        b, a = butter(2, HP / (FS / 2), btype='high')
        eeg = lfilter(b, a, eeg, axis=-1)
        eeg = eeg - eeg.mean(0, keepdims=True)
        imu_names = [c for c in names if c.startswith(('Hacc', 'Hgyro', 'Lacc', 'Lgyro', 'Racc', 'Rgyro'))]
        R = d[[names.index(c) for c in imu_names]]
        return eeg, names[:32], g[2], g[3], R
    fe = os.path.join(p.DATA, f'{sub}_scalp_{task}_{sp}.mat')
    fi = os.path.join(p.DATA, f'{sub}_IMU_{task}_{sp}.mat')
    if not (os.path.exists(fe) and os.path.exists(fi)):
        return None
    d = sio.loadmat(fe, squeeze_me=True, struct_as_record=False)
    fs0 = int(d['raw_fs'])
    clab = [str(c) for c in d['raw_clab']][:32]
    x = d['raw_x'].astype(float).T[:32]
    b, a = butter(2, HP / (fs0 / 2), btype='high')
    x = lfilter(b, a, x, axis=-1)
    x = x - x.mean(0, keepdims=True)
    b, a = butter(4, 40 / (fs0 / 2), btype='low')
    x = lfilter(b, a, x, axis=-1)[:, ::fs0 // FS]                     # 因果抗混叠后抽样
    on = np.round(np.asarray(d['event'].time, float) / 1000 * FS).astype(int)
    y = np.asarray(d['event'].y)
    di = sio.loadmat(fi, squeeze_me=True, struct_as_record=False)
    fsi = int(di['raw_fs'])
    ri = di['raw_x'].astype(float).T[p.IMU_USE]
    t_src = np.arange(ri.shape[1]) / fsi
    t_dst = np.arange(x.shape[1]) / FS
    idx = np.searchsorted(t_src, t_dst, side='right') - 1              # 零阶保持:取不晚于当前时刻的最后一个采样(严格因果,延迟 ≤ 7.8 ms)
    idx = np.clip(idx, 0, ri.shape[1] - 1)
    R = ri[:, idx]
    return x, clab, on, y, R


def running_norm(R, tau=10.0, init=5.0):
    """指数遗忘的滑动均值/方差归一化(只用过去)。"""
    a = np.exp(-1 / (tau * FS))
    n0 = int(init * FS)
    mu = R[:, :n0].mean(1)
    var = R[:, :n0].var(1) + 1e-9
    Y = np.empty_like(R)
    for t in range(R.shape[1]):
        if t >= n0:
            mu = a * mu + (1 - a) * R[:, t]
            var = a * var + (1 - a) * (R[:, t] - mu) ** 2
        Y[:, t] = (R[:, t] - mu) / np.sqrt(var + 1e-9)
    return Y


def motion_energy(R, win=2.0):
    """头部加速度模值 0.7–4 Hz 的尾随 2 s 均方(用原始量级的惯性数据,不归一化)。"""
    b, a = butter(4, [0.7 / (FS / 2), 4.0 / (FS / 2)], btype='band')
    m = np.sqrt((R[:3] ** 2).sum(0))
    m = lfilter(b, a, m - m[:5 * FS].mean())
    w = int(win * FS)
    return np.convolve(m ** 2, np.ones(w) / w, mode='full')[:len(m)]


def trailing_gate(R, calib_energy, thr_on=50.0, thr_off=20.0, step=1.0, n_on=3, n_off=3):
    """因果门控:运动能量相对本人站立校准段中位数的倍数,每 step 秒判一次;
    连续 n_on 次 ≥ thr_on 开门,连续 n_off 次 < thr_off 关门(滞回);判定作用于其后的一秒。开头 2 s 门关。"""
    e = motion_energy(R) / max(calib_energy, 1e-30)              # 不能加绝对 epsilon:OSF 版惯性量级只有 1e-13
    T = R.shape[1]
    g = np.zeros(T)
    s = int(step * FS)
    state, cnt_on, cnt_off = 0, 0, 0
    for t0 in range(2 * FS, T, s):
        r = e[t0 - 1]
        if state == 0:
            cnt_on = cnt_on + 1 if r >= thr_on else 0
            if cnt_on >= n_on:
                state, cnt_off = 1, 0
        else:
            cnt_off = cnt_off + 1 if r < thr_off else 0
            if cnt_off >= n_off:
                state, cnt_on = 0, 0
        g[t0:t0 + s] = state
    return g


def nlms_causal(X, R, gate, mu=0.05, lag=LAG):
    """因果 NLMS:输出 Y[:, t-lag] 用 R[:, t-2lag … t]。gate=0 的点不更新也不相减。"""
    k, T = R.shape
    nl = 2 * lag + 1
    W = np.zeros((k * nl, X.shape[0]))
    Y = X.copy()
    eps = 1e-6
    for t in range(2 * lag, T):
        tt = t - lag
        if gate[tt] <= 0:
            continue
        z = R[:, t - 2 * lag:t + 1].reshape(-1)                      # 21 个时滞 × k 路
        e = X[:, tt] - z @ W
        Y[:, tt] = e
        W += (mu / (z @ z + eps)) * np.outer(z, e)
    return Y


def run_subject(sub, thr, mu):
    res = {}
    gate_stats = []
    g = causal_eeg((sub, 'ERP', 'tr'))
    clf = None
    calib_energy = None
    if g is not None:
        X, clab, on, y, R = g
        f, yy = p.erp_feats(X, clab, on, y)
        clf = LDA(solver='lsqr', shrinkage='auto').fit(f, yy)
        calib_energy = float(np.median(motion_energy(R)[5 * FS:]))
    if calib_energy is None:
        for cand in (('SSVEP', '0.0'), ('ERP', '0.0')):
            g = causal_eeg((sub,) + cand)
            if g is not None:
                calib_energy = float(np.median(motion_energy(g[4])[5 * FS:]))
                break
    if calib_energy is None:
        return sub, {}, [], None
    for sp in p.SPEEDS:
        for task in ('ERP', 'SSVEP'):
            g = causal_eeg((sub, task, sp))
            if g is None:
                continue
            X, clab, on, y, R = g
            Rn = running_norm(R)
            gate = trailing_gate(R, calib_energy, thr_on=thr, thr_off=thr * 0.4)
            moving = sp != '0.0'
            n_dec = int((X.shape[1] - 2 * FS) // FS)
            on_frac = float(gate[2 * FS:].mean()) if X.shape[1] > 2 * FS else float('nan')
            gate_stats.append({'sub': sub, 'task': task, 'speed': sp, 'moving': moving, 'gate_on_frac': on_frac, 'n_decisions': n_dec})
            Xc = nlms_causal(X, Rn, gate, mu)
            Xu = nlms_causal(X, Rn, np.ones(X.shape[1]), mu)          # 不门控
            for name, Xe in (('none', X), ('nlms_c', Xu), ('nlms_gated_c', Xc)):
                try:
                    if task == 'ERP':
                        if clf is None:
                            continue
                        f, yy = p.erp_feats(Xe, clab, on, y)
                        res[f'ERP_{sp}_{name}'] = float(roc_auc_score(yy, clf.decision_function(f)))
                    else:
                        res[f'SSVEP_{sp}_{name}'] = p.ssvep_acc(Xe, clab, on, y)
                except Exception as e:
                    print(sub, task, sp, name, 'ERR', e, flush=True)
    # 过渡实验:站立 SSVEP + 快走 SSVEP 首尾相接(同一人)
    trans = None
    g0 = causal_eeg((sub, 'SSVEP', '0.0'))
    g1 = causal_eeg((sub, 'SSVEP', '1.6'))
    if g0 is not None and g1 is not None:
        X0, clab, on0, y0, R0 = g0
        X1, _, on1, y1, R1 = g1
        X = np.concatenate([X0, X1], 1)
        Rraw = np.concatenate([R0, R1], 1)
        R = running_norm(Rraw)
        gate = trailing_gate(Rraw, calib_energy, thr_on=thr, thr_off=thr * 0.4)
        T0 = X0.shape[1]
        first_on = np.argmax(gate[T0:] > 0) if (gate[T0:] > 0).any() else -1
        Xc = nlms_causal(X, R, gate, mu)
        trans = {'standing_altered_frac': float((np.abs(Xc[:, :T0] - X0) > 1e-9).any(0).mean()),
                 'gate_on_frac_standing': float(gate[2 * FS:T0].mean()), 'gate_on_frac_walking': float(gate[T0 + 2 * FS:].mean()),
                 'switch_delay_s': float(first_on / FS) if first_on >= 0 else float('nan'),
                 'ssvep_standing_none': p.ssvep_acc(X0, clab, on0, y0), 'ssvep_standing_gated': p.ssvep_acc(Xc[:, :T0], clab, on0, y0),
                 'ssvep_walking_none': p.ssvep_acc(X1, clab, on1, y1), 'ssvep_walking_gated': p.ssvep_acc(Xc[:, T0:], clab, on1, y1)}
    print(sub, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
    return sub, res, gate_stats, trans


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    mu = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    os.makedirs(OUT, exist_ok=True)
    subs = sorted({re.match(r'(s\d+)_', os.path.basename(f)).group(1) for f in glob.glob(os.path.join(p.DATA, 's*_scalp_*.mat'))})
    subs += sorted(os.path.basename(d) for d in glob.glob(os.path.join(p.OSF, 'sub-*')) if os.path.isdir(d))
    if os.environ.get('SUBS'):
        subs = [x for x in subs if x in os.environ['SUBS'].split(',')]
    out = Parallel(n_jobs=n_jobs)(delayed(run_subject)(s, thr, mu) for s in subs)
    rows = {s: r for s, r, _, _ in out}
    gates = [g for _, _, gs, _ in out for g in gs]
    trans = {s: t for s, _, _, t in out if t}
    summ = {}
    for task in ('ERP', 'SSVEP'):
        for sp in p.SPEEDS:
            for m in ('none', 'nlms_c', 'nlms_gated_c'):
                v = [r[f'{task}_{sp}_{m}'] for r in rows.values() if f'{task}_{sp}_{m}' in r]
                if v:
                    summ[f'{task}_{sp}_{m}'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)}
    # 门控混淆:以段速度为真值,逐秒决策
    tp = sum(g['gate_on_frac'] * g['n_decisions'] for g in gates if g['moving'])
    fn = sum((1 - g['gate_on_frac']) * g['n_decisions'] for g in gates if g['moving'])
    fp = sum(g['gate_on_frac'] * g['n_decisions'] for g in gates if not g['moving'])
    tn = sum((1 - g['gate_on_frac']) * g['n_decisions'] for g in gates if not g['moving'])
    gate_summary = {'per_second_decisions': {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn},
                    'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
                    'segments_moving_with_gate_on_frac_lt_0.9': [f"{g['sub']}/{g['task']}/{g['speed']}:{g['gate_on_frac']:.2f}" for g in gates if g['moving'] and g['gate_on_frac'] < 0.9],
                    'segments_standing_with_gate_on_frac_gt_0.1': [f"{g['sub']}/{g['task']}/{g['speed']}:{g['gate_on_frac']:.2f}" for g in gates if not g['moving'] and g['gate_on_frac'] > 0.1]}
    tsum = {}
    if trans:
        for k in next(iter(trans.values())):
            v = np.array([t[k] for t in trans.values()], float)
            tsum[k] = {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)), 'n': int(np.isfinite(v).sum())}
    json.dump({'per_subject': rows, 'summary': summ, 'gate': gate_summary, 'gate_segments': gates, 'transition': trans, 'transition_summary': tsum,
               'params': {'thr': thr, 'mu': mu, 'hp': HP}}, open(os.path.join(OUT, f'results_thr{thr:g}_mu{mu:g}' + ('' if HP == 0.5 else f'_hp{HP:g}') + os.environ.get('OUT_TAG', '') + '.json'), 'w'), indent=1)
    print('\n=== 因果回放 汇总 ===')
    for task in ('ERP', 'SSVEP'):
        print(task, '        none    nlms_c  nlms_gated_c')
        for sp in p.SPEEDS:
            cells = [summ.get(f'{task}_{sp}_{m}', {}).get('mean', float('nan')) for m in ('none', 'nlms_c', 'nlms_gated_c')]
            n = summ.get(f'{task}_{sp}_none', {}).get('n', 0)
            print(f'  {sp} m/s ' + '  '.join(f'{c:7.3f}' if task == 'ERP' else f'{c:7.1f}' for c in cells) + f'  (n={n})')
    print('门控逐秒决策:', {k: round(v, 1) for k, v in gate_summary['per_second_decisions'].items()}, '灵敏度 %.3f 特异度 %.3f' % (gate_summary['sensitivity'], gate_summary['specificity']))
    print('行走段门开比例<0.9:', gate_summary['segments_moving_with_gate_on_frac_lt_0.9'][:10])
    print('站立段门开比例>0.1:', gate_summary['segments_standing_with_gate_on_frac_gt_0.1'][:10])
    print('过渡实验:', {k: round(v['mean'], 3) for k, v in tsum.items()})


if __name__ == '__main__':
    main()
