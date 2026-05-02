#!/bin/bash
# SEP Pipeline — Supports QA datasets (squad, trivia_qa, nq, bioasq) and XSum.
#
# XSum is the recommended dataset for the Lookback Gate experiment because
# its long-form outputs (50-100 tokens) give the Lookback Ratio enough
# generation steps to detect and correct attention drift.  QA answers are
# 1-5 tokens — the gate fires too late to have any effect.
#
# Usage:
#   nohup bash run_pipeline.sh > pipeline_qa.log 2>&1 &          # all QA datasets
#   nohup bash run_pipeline.sh xsum > pipeline_xsum.log 2>&1 &   # XSum only
#   bash run_pipeline.sh squad trivia_qa
set -e

PYTHON=/home/gssc/.pyenv/shims/python
WORKDIR=/home/gssc/yaawar/llm/SEP_Llama-3
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_TOKEN="$HF_TOKEN"
cd "$WORKDIR"

# If specific datasets passed as args, use those; otherwise use all 4 QA datasets
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("squad" "trivia_qa" "nq" "bioasq")
fi

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

# ---- RAGTruth early-exit: stages 1+2 are handled by map_ragtruth_labels.py ----
# If the only dataset requested is ragtruth, run the label mapper then jump
# straight to stage 3.  If ragtruth is mixed with other datasets, the mapper
# runs first and the other datasets proceed through the normal stages 1-2.
RAGTRUTH_ONLY=false
if [ "${DATASETS[*]}" = "ragtruth" ]; then
    RAGTRUTH_ONLY=true
fi

if [[ " ${DATASETS[*]} " == *" ragtruth "* ]]; then
    echo "[$(date)] RAGTruth: running label mapper (replaces stages 1+2)..."
    $PYTHON map_ragtruth_labels.py 2>&1 | tee "$LOG_DIR/ragtruth_labels.log"
    echo "[$(date)] RAGTruth label mapping done."
fi

# Remove ragtruth from the list for stages 1-2 (generation + NLI)
NON_RAGTRUTH_DATASETS=()
for ds in "${DATASETS[@]}"; do
    [ "$ds" != "ragtruth" ] && NON_RAGTRUTH_DATASETS+=("$ds")
done

echo "=========================================="
echo "SEP QA Pipeline"
echo "=========================================="
echo "Datasets:   ${DATASETS[*]}"
echo "Python:     $PYTHON"
echo "Start time: $(date)"
echo "=========================================="

# ---- Stage 1: Generation (non-RAGTruth only) ----
echo ""
echo "=== STAGE 1: Generation ==="
for ds in "${NON_RAGTRUTH_DATASETS[@]}"; do
    echo "[$(date)] Starting generation for $ds..."
    $PYTHON run_qa_generation.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/gen_${ds}.log"
    echo "[$(date)] Finished generation for $ds."
    echo ""
done

# ---- Stage 2: NLI Labels (non-RAGTruth only) ----
echo "=== STAGE 2: NLI Labels ==="
for ds in "${NON_RAGTRUTH_DATASETS[@]}"; do
    echo "[$(date)] Starting NLI for $ds..."
    $PYTHON compute_nli_labels.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/nli_${ds}.log"
    echo "[$(date)] Finished NLI for $ds."
    echo ""
done

# ---- Stage 3: Feature Extraction (all datasets including ragtruth) ----
echo "=== STAGE 3: Feature Extraction ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Extracting hidden-state features for $ds..."
    $PYTHON extract_all_layers.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/extract_${ds}.log"
    echo "[$(date)] Finished extraction for $ds."
done

# ---- Stage 4: ID Evaluation ----
# STRATEGY controls which probe strategy is trained and saved.
# STRATEGIES is an ordered list: the last one whose probe is saved becomes the
# active probe used by stage 6 (inference_with_gate.py).  Default: all four.
# Override: STRATEGY=concat bash run_pipeline.sh
#           STRATEGIES="hard_vote soft_vote" bash run_pipeline.sh squad
STRATEGY=${STRATEGY:-""}
if [ -n "$STRATEGY" ]; then
    STRATEGIES=("$STRATEGY")
else
    STRATEGIES=("concat" "hard_vote" "soft_vote" "meta")
fi
TOP_K=${TOP_K:-10}

echo ""
echo "=== STAGE 4: In-Distribution Evaluation ==="
echo "Strategies: ${STRATEGIES[*]}  top_k=$TOP_K"
for ds in "${DATASETS[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        top_k_arg="--top_k $TOP_K"
        # meta uses all layers — top_k is ignored but harmless to pass
        echo "[$(date)] Training probe for $ds  strategy=$strat ..."
        $PYTHON train_probe.py \
            --mode id \
            --dataset "$ds" \
            --strategy "$strat" \
            $top_k_arg \
            --save_probe \
            2>&1 | tee "$LOG_DIR/probe_id_${ds}_${strat}.log"
        echo ""
    done
done

# ---- Stage 5: Cross-Dataset OOD Matrix ----
echo "=== STAGE 5: Cross-Dataset AUROC Matrix ==="
echo "[$(date)] Hidden-state matrix..."
$PYTHON train_probe.py --mode matrix 2>&1 | tee "$LOG_DIR/probe_matrix.log"


# ---- Stage 6: SEP-Triggered Lookback Gated Inference ----
echo ""
echo "=== STAGE 6: Gated Inference (SEP + Lookback Ratio) ==="
ALPHA=${ALPHA:-10.0}
SEP_THRESHOLD=${SEP_THRESHOLD:-0.5}
TOKEN_TYPE=${TOKEN_TYPE:-TBG}
HARD_GATE=${HARD_GATE:-""}   # set to "--hard_gate" to enable binary gate
for ds in "${DATASETS[@]}"; do
    # XSum needs more tokens and a lower accuracy threshold (handled inside
    # inference_with_gate.py automatically, but we log the intent here).
    echo "[$(date)] Running gated inference for $ds  (alpha=$ALPHA, threshold=$SEP_THRESHOLD, token=$TOKEN_TYPE)..."
    $PYTHON inference_with_gate.py \
        --dataset "$ds" \
        --alpha "$ALPHA" \
        --sep_threshold "$SEP_THRESHOLD" \
        --token_type "$TOKEN_TYPE" \
        $HARD_GATE \
        2>&1 | tee "$LOG_DIR/gated_inference_${ds}.log"
    echo ""
done

# ---- Stage 7: Causal Validation ----
echo ""
echo "=== STAGE 7: Causal Validation (Knockout & Blindness Tests) ==="
CAUSAL_SAMPLES=${CAUSAL_SAMPLES:-100}
LR_CUTOFF=${LR_CUTOFF:-0.5}
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Running causal validation for $ds  (samples=$CAUSAL_SAMPLES, lr_cutoff=$LR_CUTOFF)..."
    $PYTHON causal_validation.py \
        --dataset "$ds" \
        --num_samples "$CAUSAL_SAMPLES" \
        --lr_cutoff "$LR_CUTOFF" \
        2>&1 | tee "$LOG_DIR/causal_validation_${ds}.log"
    echo ""
done

echo ""
echo "=========================================="
echo "Pipeline complete at $(date)"
echo "=========================================="
