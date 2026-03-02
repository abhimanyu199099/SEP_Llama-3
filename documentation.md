# Project Documentation — SEP + Lookback Gating Pipeline

## Overview

This project implements **Semantic Entropy Probes (SEP)** for detecting hallucinations in Llama-2-7b-chat, extended with a **Lookback Ratio gate** (from Chuang et al., EMNLP 2024 "Lookback Lens") and **causal validation** tests.

The full pipeline has 7 stages:

```
Stage 1   run_qa_generation.py      →  generate answers + extract hidden states
Stage 2   compute_nli_labels.py     →  compute semantic entropy via NLI
Stage 3   extract_all_layers.py     →  assemble per-layer feature tensors
Stage 4   train_probe.py            →  train SEP probe, optionally save it
Stage 5   train_probe.py --mode matrix  →  cross-dataset OOD AUROC matrix
Stage 6   inference_with_gate.py    →  SEP-triggered Lookback Gated inference  [NEW]
Stage 7   causal_validation.py      →  causal knockout + blindness tests       [NEW]
```

---

## File-by-File Reference

### `common_utils.py` — Shared Constants (UNCHANGED)

Defines all shared configuration used across every script.

| Constant | Value | Meaning |
|---|---|---|
| `MODEL_NAME` | `"Llama-2-7b-chat"` | HuggingFace model identifier |
| `QA_DATASETS` | `["squad","trivia_qa","nq","bioasq"]` | The 4 datasets used throughout |
| `OUTPUT_BASE` | `"output"` | Root folder for all saved files |
| `NUM_SAMPLES_QA` | `2000` | Samples per dataset |
| `NUM_GENERATIONS_QA` | `10` | High-temp generations per question for SE |
| `TEMPERATURE_HIGH` | `1.0` | Temperature for the 10 diverse generations |
| `TEMPERATURE_LOW` | `0.1` | Temperature for the single "most likely" answer |
| `MAX_NEW_TOKENS` | `50` | Max tokens generated per answer |
| `NUM_FEW_SHOT` | `5` | Number of few-shot examples in prompt |

---

### `run_qa_generation.py` — Stage 1: Answer Generation (MODIFIED)

**What it does:**
For each question in a dataset it:
1. Generates 10 diverse answers at `TEMPERATURE_HIGH=1.0` (for computing semantic entropy).
2. Generates 1 "most likely" answer at `TEMPERATURE_LOW=0.1` with `return_latent=True` so that hidden states (TBG and SLT embeddings) are captured.
3. Saves everything to `output/{dataset}/generations.pkl`.

**Modification made:**
All calls to `model.predict()` were updated to unpack a **4-tuple**:
```python
# Before (old)
answer, log_likelihoods = model.predict(...)

# After (new)
answer, log_likelihoods, hidden_states, lookback_ratio = model.predict(...)
```
The `lookback_ratio` slot is `None` here because `return_attention=False` by default.

**Output file:** `output/{dataset}/generations.pkl`

Each entry is a dict:
```python
{
  'sample_index':      int,
  'question':          str,
  'answers':           list[str],          # ground-truth answers
  'prompt_used':       str,                # full few-shot prompt fed to the model
  'generations':       list[str],          # 10 high-temp generated answers
  'most_likely_answer': str,               # 1 low-temp answer
  'accuracy':          float,              # F1-based accuracy of most_likely_answer
  'tbg_embedding':     Tensor,             # shape (num_layers, 1, hidden_dim)  — TBG token hidden state
  'slt_embedding':     Tensor,             # shape (num_layers, 1, hidden_dim)  — SLT token hidden state
}
```

> **TBG** = "To Be Generated" token = the hidden state at the last prompt token (what the model "thinks" before answering).  
> **SLT** = "Second Last Token" = hidden state one position before TBG.

---

### `compute_nli_labels.py` — Stage 2: Semantic Entropy (UNCHANGED)

**What it does:**
Reads `generations.pkl` and uses a DeBERTa NLI model (`microsoft/deberta-v2-xlarge-mnli`) to cluster the 10 generated answers by semantic equivalence (bidirectional entailment → same cluster). Semantic Entropy = entropy over the cluster distribution.

**Output file:** `output/{dataset}/nli_labels.json`

Each entry:
```json
{"sample_index": 0, "question": "...", "entropy": 1.23, "num_clusters": 3}
```

---

### `extract_all_layers.py` — Stage 3: Feature Assembly (UNCHANGED)

**What it does:**
Merges `generations.pkl` + `nli_labels.json` into a single tensor file per dataset, stacking the TBG and SLT embeddings across all layers for all samples.

**Output file:** `output/{dataset}/all_layers.pt`
```python
{
  'X_tbg':   Tensor,   # shape (N, num_layers, hidden_dim) — TBG features
  'X_slt':   Tensor,   # shape (N, num_layers, hidden_dim) — SLT features
  'entropy': Tensor,   # shape (N,)                        — semantic entropy scores
}
```

---

### `train_probe.py` — Stage 4: Probe Training (MODIFIED)

**What it does:**
Trains a `sklearn.LogisticRegression` classifier to predict whether a sample has **high semantic entropy** (i.e., the model is uncertain/hallucinating) from its TBG or SLT hidden states.

It sweeps over all layer ranges, picks the best contiguous range by AUROC, and reports ID and OOD performance.

**Modification made — `--save_probe` flag:**

Running with `--save_probe` (only in `--mode id`) causes the script to:
1. After evaluating the probe (train/test split), **retrain on the full dataset** (no split — all N samples).
2. Save two probe bundles to disk for use by `inference_with_gate.py`.

**Saved probe files:** `output/{dataset}/sep_probe_TBG.pkl` and `output/{dataset}/sep_probe_SLT.pkl`

Each probe bundle is:
```python
{
  'clf':        LogisticRegression,  # fitted on full dataset
  'r_start':    int,                 # best layer range start (inclusive)
  'r_end':      int,                 # best layer range end (exclusive)
  'threshold':  float,               # SE binarization threshold (SE > threshold → "High SE")
  'token_type': str,                 # 'TBG' or 'SLT'
  'dataset':    str,                 # e.g. 'squad'
  'hidden_dim': int,                 # hidden dim of the model
  'num_layers': int,                 # total number of layers
}
```

The **feature vector** the probe classifier expects at inference time is:
```
np.concatenate([embedding[l] for l in range(r_start, r_end)], axis=0)
```
i.e., the TBG (or SLT) hidden state vectors from each layer in `[r_start, r_end)` concatenated into one long vector.

---

### `inference_with_gate.py` — Stage 6: Gated Inference (NEW)

**What it does:**
This is the core new pipeline. For each sample it:
1. Uses the saved SEP probe to **score the sample** from the pre-computed TBG embedding in `generations.pkl` — no LLM re-run needed for this step.
2. If `sep_score > sep_threshold` → **TRIGGER**: re-generate the answer with the Lookback Gate active.
3. If `sep_score ≤ sep_threshold` → **PASSTHROUGH**: reuse the existing `most_likely_answer`.
4. Compares accuracy before and after gating.

**The Lookback Gate (how it works):**

The gate is applied via PyTorch forward hooks placed on the attention layers in the **upper third** of the network (approximately layers 21–31 for Llama-2-7b).

For each generated token, per head in those layers:
```
Lookback Ratio (LR) = sum(attention on prompt tokens) / sum(attention on all tokens)
```
Then:
```
gate = sigmoid(LR × alpha)
```

Heads with **low LR** (attending more to newly generated text than the prompt) are suppressed. Heads with **high LR** (strongly attending to the context/prompt) are kept.

The gate is applied to the **pre-`o_proj` activations** per head, before the output projection mixes them together.

**Hook design:**
- **Hook-A** (forward hook on `self_attn`): caches the attention weight tensor `(B, H, q, kv)` for this step.
- **Hook-B** (forward_pre_hook on `o_proj`): reads the cached attention weights, computes LR per head, computes the gate, and re-scales the `x` tensor before it enters `o_proj`.

**Important rule:** The gate is **skipped** when `q_len > 1` (i.e., during the initial full-prompt pass when building the KV cache). It only fires at `q_len=1` individual token generation steps.

**CLI usage:**
```bash
python inference_with_gate.py --dataset squad
python inference_with_gate.py --dataset trivia_qa --alpha 15.0 --sep_threshold 0.6 --token_type TBG
python inference_with_gate.py --dataset squad --layer_range 21,32
```

**Arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | (required) | One of `squad trivia_qa nq bioasq` |
| `--token_type` | `TBG` | Which probe to use — TBG or SLT |
| `--sep_threshold` | `0.5` | SEP score above which gating fires |
| `--alpha` | `10.0` | Gate sharpness: higher = closer to hard binary gate |
| `--layer_range` | upper third | e.g. `21,32` → layers 21 to 31 |

**Input files required:**
- `output/{dataset}/generations.pkl` (from Stage 1)
- `output/{dataset}/sep_probe_{token_type}.pkl` (from Stage 4 with `--save_probe`)

**Output file:** `output/{dataset}/gated_results.pkl`

Each entry is the original generation dict with these fields added:
```python
{
  ...original fields...,
  'gated_answer':   str,    # answer produced with gate (or original if passthrough)
  'sep_score':      float,  # SEP uncertainty score ∈ [0,1]
  'gate_triggered': bool,   # True if SEP fired and we re-generated
}
```

---

### `causal_validation.py` — Stage 7: Causal Validation (NEW)

**What it does:**
Runs two causal experiments to prove that the gate mechanism is causally responsible for accuracy improvements, not just coincidence.

**Test 1 — Knockout Test:**

*Hypothesis:* The sigmoid gate is better than doing nothing (original) AND better than hard zeroing (knockout). If completely zeroing ALL upper-layer heads hurts accuracy more than the sigmoid gate, it proves those heads are causally involved and that the soft sigmoid gate is the right intervention.

*On:* triggered samples (samples where SEP fired in Stage 6).

*What it measures:*
```
original_acc   →  accuracy of most_likely_answer before gating
gated_acc      →  accuracy of sigmoid-gated re-generation
knockout_acc   →  accuracy when ALL upper-layer heads are hard-zeroed
```

Expected result: `gated_acc > original_acc` and `gated_acc > knockout_acc`.

**Test 2 — Blindness Test:**

*Hypothesis:* Heads with HIGH Lookback Ratio (strongly attending to the prompt) are the "grounding" heads responsible for factually correct answers. Suppressing them should degrade accuracy.

*On:* passthrough samples (samples where SEP did NOT fire — model was confident and correct).

*What it measures:*
```
original_acc   →  accuracy of existing answer
blindness_acc  →  accuracy when high-LR heads (LR ≥ lr_cutoff) are zeroed
```

Expected result: `blindness_acc < original_acc` (grounding heads, when removed, hurt accuracy).

**CLI usage:**
```bash
python causal_validation.py --dataset squad
python causal_validation.py --dataset trivia_qa --num_samples 50 --lr_cutoff 0.4
```

**Arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | (required) | One of `squad trivia_qa nq bioasq` |
| `--token_type` | `TBG` | Probe to use |
| `--num_samples` | `100` | How many triggered/passthrough samples to test per condition |
| `--lr_cutoff` | `0.5` | LR threshold for blindness test (heads with LR ≥ this are zeroed) |
| `--layer_range` | upper third | Same as inference_with_gate.py |

**Input files required:**
- `output/{dataset}/gated_results.pkl` (from Stage 6)
- `output/{dataset}/sep_probe_{token_type}.pkl` (from Stage 4 with `--save_probe`)

**Output file:** `output/{dataset}/causal_validation.pkl`
```python
{
  'knockout_results':  list[dict],  # one dict per test-1 sample
  'blindness_results': list[dict],  # one dict per test-2 sample
  'config':            dict,        # args used
}
```

---

### `semantic_uncertainty/uncertainty/models/huggingface_models.py` — Model Wrapper (MODIFIED)

**Modification:** `predict()` now returns a **4-tuple** instead of a 2-tuple:
```python
return sliced_answer, log_likelihoods, hidden_states, lookback_ratio
```

The new parameters:
- `return_latent=True` → fills `hidden_states` with TBG/SLT embeddings (used in Stage 1).
- `return_attention=False` (default) → `lookback_ratio` is `None`. If `True`, attention weights are returned from `model.generate()` and LR per head is computed.

All call sites across the codebase were updated to unpack all 4 values (using `_` for unused slots).

---

### `run_pipeline.sh` — Full Pipeline Script (MODIFIED)

Runs all 7 stages sequentially across all 4 datasets.

**New stages added:**
- Stage 4 now uses `--save_probe` flag.
- Stage 6 added: `inference_with_gate.py` with configurable `ALPHA`, `SEP_THRESHOLD`, `TOKEN_TYPE`.
- Stage 7 added: `causal_validation.py` with configurable `CAUSAL_SAMPLES`, `LR_CUTOFF`.

**Environment variables you can set before running:**
```bash
ALPHA=10.0           # gate sharpness (default: 10.0)
SEP_THRESHOLD=0.5    # SEP trigger threshold (default: 0.5)
TOKEN_TYPE=TBG       # probe type (default: TBG)
CAUSAL_SAMPLES=100   # samples for causal tests (default: 100)
LR_CUTOFF=0.5        # blindness test cutoff (default: 0.5)
```

---

## Step-by-Step Execution Guide

### Prerequisites

```bash
conda activate se_probes
cd /home/anish/yaawar/LLM/semantic-entropy-probes
```

### Step 1 — Generate answers and extract hidden states
```bash
python run_qa_generation.py --dataset squad
# Repeat for: trivia_qa, nq, bioasq
# Output: output/squad/generations.pkl
```

### Step 2 — Compute NLI-based semantic entropy labels
```bash
python compute_nli_labels.py --dataset squad
# Output: output/squad/nli_labels.json
```

### Step 3 — Assemble per-layer feature tensors
```bash
python extract_all_layers.py --dataset squad
# Output: output/squad/all_layers.pt
```

### Step 4 — Train SEP probe and save it
```bash
python train_probe.py --mode id --dataset squad --save_probe
# Output: output/squad/sep_probe_TBG.pkl
#         output/squad/sep_probe_SLT.pkl
```

> The `--save_probe` flag retrains the probe on the **full dataset** (no train/test split) after evaluation and saves it. This is needed by Stage 6.

### Step 5 — (Optional) OOD cross-dataset matrix
```bash
python train_probe.py --mode matrix
# Shows 4×4 AUROC table
```

### Step 6 — Run gated inference
```bash
python inference_with_gate.py --dataset squad --alpha 10.0 --sep_threshold 0.5
# Output: output/squad/gated_results.pkl
# Prints accuracy comparison: original vs gated
```

### Step 7 — Run causal validation
```bash
python causal_validation.py --dataset squad --num_samples 100 --lr_cutoff 0.5
# Output: output/squad/causal_validation.pkl
# Prints Test 1 (knockout) and Test 2 (blindness) results
```

### All at once (all 4 datasets)
```bash
ALPHA=10.0 SEP_THRESHOLD=0.5 TOKEN_TYPE=TBG CAUSAL_SAMPLES=100 bash run_pipeline.sh
```

---

## Output File Tree

After running all stages for one dataset:
```
output/
  squad/
    generations.pkl          ← Stage 1: answers, embeddings, prompts
    nli_labels.json          ← Stage 2: entropy per sample
    all_layers.pt            ← Stage 3: X_tbg, X_slt, entropy tensors
    sep_probe_TBG.pkl        ← Stage 4 (--save_probe): trained TBG probe bundle
    sep_probe_SLT.pkl        ← Stage 4 (--save_probe): trained SLT probe bundle
    gated_results.pkl        ← Stage 6: per-sample gated answers + sep_score
    causal_validation.pkl    ← Stage 7: knockout + blindness test results
  logs/
    gen_squad.log
    nli_squad.log
    extract_squad.log
    probe_id_squad.log
    gated_inference_squad.log
    causal_validation_squad.log
```

---

## Known Artifacts (Safe to Ignore)

These files/functions were created during an earlier incorrect implementation and are **not part of the working pipeline**. They do not affect the normal pipeline (Stages 1–7).

| Artifact | Location | Status |
|---|---|---|
| `extract_lookback_features.py` | root | Wrong-approach file — precomputed LR offline. Not called anywhere in the pipeline. |
| `main_lookback_id/ood/matrix()` | `train_probe.py` | Leftover functions, only run if `--feature_type lookback` is passed explicitly — never called by `run_pipeline.sh`. |

`run_pipeline.sh` no longer references any of these (Stage 3b and all `--feature_type lookback` calls have been removed).

---

## Key Concepts

### What is Semantic Entropy (SE)?
SE measures how "spread out" the 10 generated answers are semantically. If all 10 answers say the same thing → low SE (confident). If answers disagree → high SE (uncertain/hallucinating).

### What is the SEP Probe?
A logistic regression classifier that learns to predict high vs low SE from just the hidden state at the TBG or SLT token at inference time — before generating any answer. This means you can flag uncertainty **before** you generate, at low compute cost.

### What is Lookback Ratio?
For each attention head at each layer, LR measures what fraction of that head's attention goes to the input prompt (context) vs the tokens it has already generated:
```
LR[head] = sum(attn_to_prompt_tokens) / sum(attn_to_all_tokens)
```
High LR → head is "grounded" in the source context.  
Low LR → head is attending to its own previous outputs (riskier for hallucination).

### How does the gate work?
```
gate[head] = sigmoid(LR[head] × alpha)
```
- If `alpha=10` and `LR=0.3`: `sigmoid(3) ≈ 0.95` → head mostly kept.
- If `alpha=10` and `LR=0.05`: `sigmoid(0.5) ≈ 0.62` → head partially suppressed.
- If `alpha=10` and `LR=0.0`: `sigmoid(0) = 0.5` → head halved.

The gate is applied to the pre-`o_proj` activations: it re-scales each head's output vector before the output projection combines all heads.

### Why upper layers only?
Research (Chuang et al. 2024, others) shows that factual recall and grounding decisions happen primarily in the upper third of transformer layers. Gating lower layers risks disrupting syntax/language modeling; gating upper layers targets the hallucination-prone decision heads.
