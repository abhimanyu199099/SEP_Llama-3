#!/bin/bash
# Runs stages 6-8 for QA datasets or XSum.
# For the Lookback Gate experiment, prefer XSum (long-form output means
# the gate fires meaningfully across many tokens, unlike 1-5 token QA answers).
#
# Usage:
#   bash run_remaining.sh                    # QA datasets (squad, trivia_qa, nq)
#   DATASETS=(xsum) bash run_remaining.sh    # XSum only
set -e
PYTHON=/home/anish/miniconda3/envs/se_probes/bin/python
WORKDIR=/home/anish/yaawar/LLM/semantic-entropy-probes
cd "$WORKDIR"

DATASETS=("squad" "trivia_qa" "nq")
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Remaining Pipeline: Gated Inference + Causal Validation + Lookback"
echo "Start: $(date)"
echo "=========================================="

echo ""
echo "=== STAGE 6: Gated Inference (SEP + Lookback Ratio) ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Gated inference for $ds..."
    $PYTHON inference_with_gate.py --dataset "$ds" --alpha 10.0 --sep_threshold 0.5 --token_type TBG \
        2>&1 | tee "$LOG_DIR/gated_inference_${ds}.log"
    echo "[$(date)] Done: $ds"
done

echo ""
echo "=== STAGE 7: Causal Validation ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Causal validation for $ds..."
    $PYTHON causal_validation.py --dataset "$ds" --num_samples 100 --lr_cutoff 0.5 \
        2>&1 | tee "$LOG_DIR/causal_validation_${ds}.log"
    echo "[$(date)] Done: $ds"
done

echo ""
echo "=== STAGE 8: Lookback Feature Extraction (optional) ==="
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] Extracting lookback features for $ds..."
    $PYTHON extract_lookback_features.py --dataset "$ds" \
        2>&1 | tee "$LOG_DIR/lookback_${ds}.log"
    echo "[$(date)] Done: $ds"
done

echo ""
echo "=========================================="
echo "All remaining stages complete at $(date)"
echo "=========================================="
