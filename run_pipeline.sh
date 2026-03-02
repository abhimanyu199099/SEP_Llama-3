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

PYTHON=/home/anish/miniconda3/envs/se_probes/bin/python
WORKDIR=/home/anish/yaawar/LLM/semantic-entropy-probes
cd "$WORKDIR"

# If specific datasets passed as args, use those; otherwise use all 4 QA datasets
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("squad" "trivia_qa" "nq" "bioasq")
fi

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "SEP QA Pipeline"
echo "=========================================="
echo "Datasets:   ${DATASETS[*]}"
echo "Python:     $PYTHON"
echo "Start time: $(date)"
echo "=========================================="

# ---- Stage 1: Generation ----
echo ""
echo "=== STAGE 1: Generation ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Starting generation for $ds..."
    $PYTHON run_qa_generation.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/gen_${ds}.log"
    echo "[$(date)] Finished generation for $ds."
    echo ""
done

# ---- Stage 2: NLI Labels ----
echo "=== STAGE 2: NLI Labels ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Starting NLI for $ds..."
    $PYTHON compute_nli_labels.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/nli_${ds}.log"
    echo "[$(date)] Finished NLI for $ds."
    echo ""
done

# ---- Stage 3: Feature Extraction ----
echo "=== STAGE 3: Feature Extraction ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Extracting hidden-state features for $ds..."
    $PYTHON extract_all_layers.py --dataset "$ds" 2>&1 | tee "$LOG_DIR/extract_${ds}.log"
    echo "[$(date)] Finished extraction for $ds."
done

# ---- Stage 4: ID Evaluation ----
echo ""
echo "=== STAGE 4: In-Distribution Evaluation ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Training hidden-state ID probe for $ds..."
    $PYTHON train_probe.py --mode id --dataset "$ds" --save_probe 2>&1 | tee "$LOG_DIR/probe_id_${ds}.log"
    echo ""
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
