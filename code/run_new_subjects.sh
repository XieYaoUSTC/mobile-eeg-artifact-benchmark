#!/usr/bin/env bash
# 只对新增被试(SUBS=逗号分隔)跑完整流水线,输出文件带 _new6 后缀;随后用 merge_new_subjects.py 并入主结果文件。
# 用法: SUBS=sub-03,sub-05,... bash code/run_new_subjects.sh [并行数]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
J=${1:-4}
export SUBS
export OUT_TAG=_new6
echo "subjects: $SUBS  jobs: $J"
$PY code/p1_clean_eval.py none,asr,asr10,reg,cca,gait,gait+reg $J results_24_new6
$PY code/p1_clean_eval.py none,nlms,nlms_gated $J results_24_gated_new6
$PY code/p1_clean_eval.py none,nlms_gated,nlms_mu02,nlms_mu10,nlms_mu20,cca20,cca40 $J results_24_tuning_new6
$PY code/p1_clean_eval.py none,asr,asr10,icc,icc_w4 $J results_24_asrfix_icc_new6
$PY code/p1_clean_eval.py none,reg,reg_head,reg_ankle,nlms $J results_24_ablation_new6
$PY code/p1_clean_eval.py none,reg_shift,nlms_shift,reg_a1,reg_a3 $J results_24_controls_new6
$PY code/p1_clean_eval.py none,reg,nlms_gated,reg_shift10,reg_shift120,reg_surr,nlms_shift10,nlms_shift120,nlms_surr $J results_24_controls2_new6
$PY code/p1_clean_eval.py none,reg,reg_b10,reg_b30,cca40_nolag,cca50,cca60,cca70 $J results_24_methods2_new6
$PY code/p1_clean_eval.py none,icc_w4_lag,icc_w20,icc_w20_lag $J results_24_icc2_new6
$PY code/p1_asr_standard.py $J
$PY code/imu_only_decoding.py
for thr in 50 100 200; do $PY code/p1_causal_replay.py $J $thr 0.05; done
CAUSAL_HP=0.1 $PY code/p1_causal_replay.py $J 100 0.05
echo "ALL NEW-SUBJECT RUNS DONE"
