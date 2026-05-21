#!/bin/bash
# Resume from Stage 2.5: QTIP Qwen3-4B (K=2,3,4) → Stage 3: Qwen3-8B compress+eval.
# Stage 1 (4B baseline) and Stage 2 (4B compressed eval) already complete.
set -u

ROOT="/home/jgryu/workspace/weight_compression/comp_lm_qtip"
QTIP_ROOT="/home/jgryu/workspace/weight_compression/qtip"
NOTIFY_PY="$ROOT/scripts/_paroquant_notify.py"
RES_ROOT="../hf_model_comp_results_v2"

cd "$ROOT"

##########################################################################
## Stage 2.5: QTIP Qwen3-4B (K=2,3,4) compress + eval
##########################################################################
echo "==================== Stage 2.5: QTIP Qwen3-4B (K=2,3,4) ===================="
( cd "$QTIP_ROOT" && bash run_qtip_qwen3_4b.sh )
( cd "$QTIP_ROOT" && bash run_qtip_qwen3_4b_eval.sh )

##########################################################################
## Stage 3a: Qwen3-8B compression sweep
##########################################################################
echo "==================== Stage 3a: Qwen3-8B compression sweep ===================="
bash scripts/comp_lm_reasoning_8b.sh

##########################################################################
## Stage 3b: Qwen3-8B reasoning eval
##########################################################################
echo "==================== Stage 3b: Qwen3-8B reasoning eval ===================="
sed -i 's|^    # "Qwen--Qwen3-8B"|    "Qwen--Qwen3-8B"|' scripts/eval_reasoning_paroquant.sh
sed -i 's|^    "Qwen--Qwen3-4B"$|    # "Qwen--Qwen3-4B"  # done in Stage 2|' scripts/eval_reasoning_paroquant.sh

bash scripts/eval_reasoning_paroquant.sh

# Restore tidiness.
sed -i 's|^    # "Qwen--Qwen3-4B"  # done in Stage 2|    "Qwen--Qwen3-4B"|' scripts/eval_reasoning_paroquant.sh
sed -i 's|^    "Qwen--Qwen3-8B"$|    # "Qwen--Qwen3-8B"  # 8B compression not yet done; enable after compress sweep|' scripts/eval_reasoning_paroquant.sh

mkdir -p "log/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft"
python3 "$NOTIFY_PY" "Stage 3: Qwen3-8B 압축 (6 λ) reasoning 완료" \
    "${RES_ROOT}/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft" \
    2>&1 | tee -a "log/Qwen--Qwen3-8B/ql_ldlq128_rnorm_ft/stage3_summary.log"

echo "==================== Resume complete ===================="
