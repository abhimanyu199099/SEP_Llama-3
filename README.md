# Semantic Entropy Probes with Head-Adaptive Lookback Gating

[![arXiv](https://img.shields.io/badge/arXiv-2406.15927-b31b1b.svg)](https://arxiv.org/abs/2406.15927)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository implements **Semantic Entropy Probes (SEPs)** combined with **Lookback Ratio head-adaptive gating** for hallucination detection and causal validation in Large Language Models, using `meta-llama/Llama-2-7b-chat-hf`.

The pipeline supports multiple datasets: **SQuAD**, **TriviaQA**, **NQ**, **BioASQ** (QA), **XSum** (abstractive summarization), and **HaluEval QA** (context-grounded QA).

Adapted from [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes), extended with multi-dataset support, Lookback Ratio gating, SEP-modulated inference, and causal ablation experiments.

---

## Overview

Large Language Models hallucinate — producing fluent but factually incorrect text. This project addresses the problem at two levels:

1. **Detection**: Train a linear probe on a single-forward-pass hidden state to predict semantic uncertainty (hallucination risk) without multi-sampling at inference time.
2. **Causal Validation**: Use a Lookback Ratio gate (triggered by the probe) to demonstrate that certain attention heads *causally* drive hallucinated outputs, via knockout and blindness ablation tests.

### Key Components

- **Semantic Entropy Probe (SEP)**: Logistic regression on the TBG (Token Before Generation) or SLT (Second-to-Last Token) hidden state. Trained on semantic entropy labels derived from NLI clustering of 10 stochastic generations. Predicts hallucination risk in a single forward pass (near-zero overhead).
- **Lookback Ratio (LR)**: Per-head metric measuring how much each attention head attends to the input context vs. self-generated tokens. High LR = context-grounded; low LR = self-referential / drifting.
- **SEP-Modulated Gate**: At inference time, if the probe scores a sample as uncertain (`sep_score > threshold`), a sigmoid gate modulated by the probe score suppresses low-LR heads:
  ```
  gate_shape = sigmoid((LR - cutoff) × α)
  gate = sep_score × gate_shape + (1 − sep_score)
  ```
  This blends between no-op (when the probe is confident) and full gating (when the probe is maximally uncertain), preventing over-suppression on borderline samples.

---

## Pipeline Architecture

The pipeline runs in **7 sequential stages**:

```
Multi-dataset Inputs
        |
        v
[1. Generation]          run_qa_generation.py
        |                  - 10 stochastic answers per question (T=1.0)
        |                  - 1 greedy answer per question (T=0.1)
        |                  - Output: output/{dataset}/generations.pkl
        v
[2. NLI Labels]          compute_nli_labels.py
        |                  - DeBERTa-v2-XL NLI clustering across 10 generations
        |                  - Semantic entropy → binary hallucination label
        |                  - Output: output/{dataset}/nli_labels.pkl
        v
[3. Feature Extraction]  extract_all_layers.py
        |                  - Greedy forward pass per sample
        |                  - TBG & SLT hidden states from all 32 layers
        |                  - Output: output/{dataset}/sep_dataset_all_layers.pt
        v
[4. ID Probe Training]   train_probe.py --mode id
        |                  - Logistic regression on TBG/SLT hidden states
        |                  - 80/20 train/test split with StandardScaler
        |                  - Reports per-layer Accuracy and AUROC
        |                  - Output: output/{dataset}/sep_probe_{TBG|SLT}.pkl
        v
[5. OOD Matrix]          train_probe.py --mode matrix
        |                  - Cross-dataset AUROC: train on one, eval on all others
        |                  - Output: printed matrix + output/logs/probe_matrix.log
        v
[6. Gated Inference]     inference_with_gate.py
        |                  - Loads saved probe (Stage 4)
        |                  - Scores each sample; triggers SEP-modulated LR gate
        |                  - Compares original vs. gated answer accuracy
        |                  - Output: output/{dataset}/gated_results.pkl
        v
[7. Causal Validation]   causal_validation.py
                           - Test 1 Knockout: on SEP-triggered samples —
                             original vs. soft-gate vs. hard-zero accuracy
                           - Test 2 Blindness: on passthrough samples —
                             zero high-LR (grounding) heads → expect accuracy drop
                           - Output: output/{dataset}/causal_validation.pkl
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- GPU with at least 24 GB VRAM (FP16 inference of Llama-2-7b-chat)
- Access to [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (gated model)

### Install Dependencies

```bash
conda env update -f sep_enviroment.yaml
conda activate se_probes
```

Or manually:

```bash
pip install torch transformers datasets rouge-score scikit-learn accelerate matplotlib scipy
```

### Authenticate with Hugging Face

```bash
huggingface-cli login
export HF_HUB_CACHE=/path/to/your/hf_cache
```

---

## Usage

### Run the Full Pipeline

```bash
# All QA datasets (squad, trivia_qa, nq, bioasq)
nohup bash run_pipeline.sh > pipeline_qa.log 2>&1 &

# XSum only
nohup bash run_pipeline.sh xsum > pipeline_xsum.log 2>&1 &

# Specific datasets
bash run_pipeline.sh squad trivia_qa
```

The script waits for Stage 1 generation logs before proceeding — generation can be launched separately on multiple GPUs in parallel.

### Step-by-Step

```bash
# 1. Generate stochastic answers / summaries
python run_qa_generation.py --dataset squad
# Output: output/squad/generations.pkl

# 2. Compute NLI-based semantic entropy labels
python compute_nli_labels.py --dataset squad
# Output: output/squad/nli_labels.pkl

# 3. Extract hidden-state features (all layers)
python extract_all_layers.py --dataset squad
# Output: output/squad/sep_dataset_all_layers.pt

# 4. Train ID probe and save it
python train_probe.py --mode id --dataset squad --save_probe
# Output: output/squad/sep_probe_TBG.pkl, sep_probe_SLT.pkl

# 5. Cross-dataset OOD matrix
python train_probe.py --mode matrix
# Prints AUROC matrix for all datasets

# 6. Gated inference with SEP-modulated Lookback gate
python inference_with_gate.py --dataset squad --alpha 10.0 --sep_threshold 0.5
# Output: output/squad/gated_results.pkl

# 7. Causal validation
python causal_validation.py --dataset squad --num_samples 100
# Output: output/squad/causal_validation.pkl
```

### Generate Figures

```bash
python generate_figures.py
# Outputs poster-quality PNGs to output/figures/
```

Produces:
- **Figure 2**: Cross-dataset AUROC heatmap (OOD/ID generalization)
- **Figure 3**: SEP score distributions — Correct vs. Hallucinated
- **Figure 4**: Accuracy before/after gating (XSum + HaluEval)
- **Figure 5**: Causal validation — Original / Soft Gate / Hard Knockout per dataset

---

## Project Structure

```
SEP_Llama-3/
├── run_qa_generation.py         # Stage 1: Stochastic generation (QA + XSum)
├── compute_nli_labels.py        # Stage 2: DeBERTa NLI semantic entropy labels
├── extract_all_layers.py        # Stage 3: TBG/SLT hidden states, all 32 layers
├── train_probe.py               # Stage 4/5: ID probe training + OOD matrix
├── inference_with_gate.py       # Stage 6: SEP-triggered LR-gated inference
├── causal_validation.py         # Stage 7: Knockout + Blindness ablation tests
├── generate_figures.py          # Poster/report figure generation
├── common_utils.py              # Shared constants and dataset configs
├── run_pipeline.sh              # Orchestrates all 7 stages
├── sep_enviroment.yaml          # Conda environment
├── semantic_uncertainty/        # Core library (adapted from OATML)
│   ├── generate_answers.py
│   ├── compute_uncertainty_measures.py
│   └── uncertainty/
│       ├── models/              # HuggingFace model wrappers
│       ├── data/                # Dataset loaders (SQuAD, TriviaQA, NQ, BioASQ, HaluEval, XSum)
│       └── uncertainty_measures/  # Semantic entropy, p_ik, p_true
├── output/
│   ├── {dataset}/               # Per-dataset outputs (pkl, pt, probe pkl)
│   ├── figures/                 # Generated PNGs
│   └── logs/                    # Stage logs
└── slurm/
    └── run.sh                   # SLURM batch job script
```

---

## Datasets

| Dataset | Type | Samples | Generations | Accuracy Metric |
|---------|------|---------|-------------|-----------------|
| SQuAD | Extractive QA | 2,000 | 10 | Token-F1 ≥ 0.5 |
| TriviaQA | Open-domain QA | 2,000 | 10 | Token-F1 ≥ 0.5 |
| NQ | Open-domain QA | 2,000 | 10 | Token-F1 ≥ 0.5 |
| BioASQ | Biomedical QA | 2,000 | 10 | Token-F1 ≥ 0.5 |
| XSum | Summarization | 1,000 | 5 | ROUGE-L ≥ 0.2 |
| HaluEval QA | Context QA | 2,000 | 10 | Token-F1 ≥ 0.5 |

---

## Key Design Decisions

**Average-normalized Lookback Ratio**: The LR is computed as average per-token attention mass, not total sum. On long XSum articles (500+ prompt tokens), sum-based LR is trivially high (~0.9) and the gate never fires. Average-based LR (~0.15–0.36) makes the gate responsive.

**SEP-modulated blend**: Rather than scaling the sigmoid's sharpness (which doesn't change which heads fire), the gate shape is blended linearly with a pass-through using the probe's uncertainty score. This reduces collateral suppression on borderline samples.

**XSum as LR testbed**: Lookback Ratio requires multiple generated tokens (50–100) to observe attention drift between context and generation. QA answers are 1–5 tokens — the gate fires too late to have an effect. XSum is the correct dataset for this experiment.

**Causal validation over mitigation**: The primary contribution of the gating experiment is not accuracy improvement but *causal attribution*. Test 1 (soft gate 49% > hard knockout 42% on triggered samples) and Test 2 (zeroing grounding heads degrades passthrough accuracy) together demonstrate that the probe identifies causally relevant heads, not noise.

---

## Results Summary

| Dataset | ID AUROC | OOD/ID Ratio |
|---------|----------|--------------|
| SQuAD | ~0.70 | ~0.86 |
| TriviaQA | ~0.72 | ~0.85 |
| NQ | ~0.68 | ~0.87 |
| HaluEval | ~0.74 | ~0.86 |

*Exact values from `probe_matrix.log` after running Stage 5.*

**Causal Validation (XSum):**
- Test 1 Knockout: Original 42% → Soft Gate 49% → Hard Zero 42%
- Test 2 Blindness: Zeroing high-LR heads degrades passthrough accuracy, confirming causal role

---

## Credits

Based on:

> **Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs**
> Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal
> arXiv:2406.15927 (2024)

Original repository: [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes)

```bibtex
@misc{kossen2024semanticentropyprobesrobust,
      title={Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs},
      author={Jannik Kossen and Jiatong Han and Muhammed Razzak and Lisa Schut and Shreshth Malik and Yarin Gal},
      year={2024},
      eprint={2406.15927},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15927},
}
```
