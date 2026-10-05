"""条件精确符号秩检验(供 make_tables.py / make_supplement_tex.py 共用)。"""
import numpy as np


def exact_signed_rank_p(d, decimals=12):
    """条件精确双侧符号秩检验:给定观测到的 |d| 中秩(并列取平均秩),零差值保留并把其秩对半分到两侧(zsplit),
    只对非零差值的符号做穷举(动态规划计数),双侧 p = 2×min(两侧尾概率),上限 1。
    与 scipy.stats.wilcoxon(zero_method='zsplit') 的统计量一致;scipy 的 method='exact' 在有并列/零值时不是条件精确分布。"""
    from scipy.stats import rankdata
    d = np.round(np.asarray(d, float), decimals)
    n = len(d)
    if n == 0 or np.all(d == 0):
        return 1.0
    r = rankdata(np.abs(d))
    r2 = np.round(r * 2).astype(int)
    zero = d == 0
    base = r2[zero].sum() / 2.0
    tplus = r2[d > 0].sum() + base
    nz = r2[~zero]
    S = int(nz.sum())
    dp = np.zeros(S + 1, dtype=object); dp[0] = 1
    for v in nz:
        dp[v:] = dp[v:] + dp[:S + 1 - v]
    total = sum(dp)
    sums = np.arange(S + 1) + base
    lo = sum(dp[sums <= tplus + 1e-9]); hi = sum(dp[sums >= tplus - 1e-9])
    return float(min(1.0, 2 * min(lo, hi) / total))
