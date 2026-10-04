#!/usr/bin/env bash
# v4 收尾链:并入新被试 → 留人选参 → 一致性/探针 → 表 → 补充材料 → 图 → 编译。
# 用法: bash code/finalize_v4.sh [并行数]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
J=${1:-6}
DUP=sub-19,sub-20,sub-21,sub-22,sub-23,sub-24
NEW=sub-01,sub-02,sub-03,sub-04,sub-15,sub-16
echo "== merge"; $PY code/merge_new_subjects.py $DUP $NEW
echo "== held-out params"; $PY code/heldout_param_selection.py
echo "== held-out gate"; $PY code/heldout_gate_threshold.py
echo "== release concordance"; $PY code/release_concordance.py
echo "== ssvep probe"; $PY code/ssvep_signal_probe.py $J
echo "== tables"; $PY code/make_tables.py > /dev/null
echo "== supplement"; $PY code/make_supplement_tex.py
echo "== figures"; $PY code/make_figures_en.py
echo "== compile"
cd paper_jne
pdflatex -interaction=nonstopmode supplement.tex > /dev/null; pdflatex -interaction=nonstopmode supplement.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null
grep -E "^!|Output written" main.log supplement.log | sort | uniq -c
echo "FINALIZE DONE"
