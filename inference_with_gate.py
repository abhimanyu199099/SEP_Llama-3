"""SEP-Triggered Lookback Gating — Inference Script (v2)

Three structural fixes applied vs. the original:

  FIX 1 — Off-by-One Sync (Architectural)
    Replaced the two-hook design (forward_hook on self_attn +
    forward_pre_hook on o_proj) with per-instance monkey-patching of
    LlamaAttention.forward.  The patched forward intercepts attn_weights
    right after softmax and gates attn_output BEFORE o_proj, all inside one
    atomic forward call.  Token t is now gated using LR(t), not LR(t-1).

  FIX 2 — Mute Bug (Mathematical)
    Old gate: sigmoid(LR × α).  Because LR ∈ [0,1], a maximally-hallucinating
    head (LR=0) got gate=0.5 and a grounding head (LR=1) got gate~0.999.
    This uniformly shrinks the residual stream → RMSNorm amplifies noise.
    New gate: sigmoid((LR − cutoff) × α).  This centres the gate on
    lr_cutoff so hallucinating heads (LR→0) are suppressed toward 0 and
    grounding heads (LR→1) pass through near 1.  Optional hard-gate:
    (LR ≥ cutoff).float().

  FIX 3 — XSum Dataset (Conceptual)
    Lookback Ratio only fires when the model has generated ≥1 token from the
    context window.  QA answers are 1-5 tokens → the gate almost never
    triggers before the answer is committed.  XSum summaries are 50-100
    tokens → attention drift can be detected and corrected mid-generation.
    Added --dataset xsum with 100-token budget and ROUGE-L scoring.

Pipeline (per sample):
  1. Load pre-computed TBG embedding from generations.pkl.
  2. SEP probe scores the embedding → uncertainty ∈ [0, 1].
  3. If uncertainty > sep_threshold → TRIGGER:
       - LookbackGatedAttention is live on upper LLM layers.
       - Re-generate; each decode step computes per-head Lookback Ratio
         and applies gate = σ((LR − cutoff) × α) before o_proj.
       - output_attentions is NOT needed: the monkey-patched forward
         computes attention weights internally.
  4. If uncertainty ≤ sep_threshold → passthrough (reuse existing answer).
  5. Compute accuracy (ROUGE-L for XSum, token-F1 for QA) and compare.

Saved to: output/{dataset}/gated_results.pkl

Usage:
    python inference_with_gate.py --dataset xsum
    python inference_with_gate.py --dataset xsum --alpha 15.0 --gate_mode hard
    python inference_with_gate.py --dataset squad --lr_cutoff 0.4
    python inference_with_gate.py --dataset trivia_qa --sep_threshold 0.6
"""
import math
import os
import sys
import gc
import pickle
import logging
import argparse
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "semantic_uncertainty"))

from uncertainty.models.huggingface_models import HuggingfaceModel, StoppingCriteriaSub
from transformers import StoppingCriteriaList
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from common_utils import (
    MODEL_NAME, ALL_DATASETS, XSUM_DATASETS, OUTPUT_BASE,
    MAX_NEW_TOKENS, XSUM_MAX_NEW_TOKENS, XSUM_ACC_THRESHOLD,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# ------------------------------------------------------------------ #
# LookbackGatedAttention                                               #
#                                                                      #
# Replaces LlamaAttention.forward on target layers via per-instance    #
# monkey-patching.  Fixes the off-by-one sync bug: attn_weights and    #
# o_proj input are intercepted atomically in a single forward call.    #
# ------------------------------------------------------------------ #

class LookbackGatedAttention:
    """Gates Lookback-Ratio-identified heads by monkey-patching each target
    LlamaAttention instance's .forward method rather than using two separate
    hooks (which caused token t to be gated by LR(t-1)).

    The patched forward is a full re-implementation of LlamaAttention.forward
    (verbatim for transformers==4.35.2) with one extra block between softmax
    and o_proj.  When self.triggered is False or q_len > 1 (prompt processing),
    it falls through to the original bound method — zero overhead.

    Gate modes (Fix 2):
        'soft'      → gate = σ((LR − cutoff) × α)         [default]
        'hard'      → gate = (LR ≥ cutoff).float()
        'zero_all'  → gate = 0 for all heads               [causal knockout]
        'zero_high' → gate = 0 where LR ≥ cutoff, else 1   [blindness test]
        'zero_low'  → gate = 0 where LR < cutoff, else 1   [hard suppress]
    """

    def __init__(self, model, context_length=0, alpha=10.0, lr_cutoff=0.5,
                 gate_mode='soft', layer_range=None):
        self.triggered      = False
        self.context_length = context_length
        self.alpha          = alpha
        self.lr_cutoff      = lr_cutoff
        self.gate_mode      = gate_mode
        self._patched       = []   # list of (attn_mod, original_bound_method)

        layers = model.model.layers
        n = len(layers)
        if layer_range is None:
            layer_range = range(n * 2 // 3, n)

        for idx in layer_range:
            attn_mod = layers[idx].self_attn
            original = attn_mod.forward
            attn_mod.forward = self._make_gated_forward(original)
            self._patched.append((attn_mod, original))

        logging.info(
            f"LookbackGatedAttention: patched {len(self._patched)} layers "
            f"({list(layer_range)[0]}–{list(layer_range)[-1]}), "
            f"mode={gate_mode}, alpha={alpha}, lr_cutoff={lr_cutoff}"
        )

    # ---- Patched forward factory ----------------------------------- #

    def _make_gated_forward(self, original_bound):
        """Return a replacement for attn_mod.forward.

        The closure captures ``original_bound`` (to fall through cleanly) and
        ``self`` (the controller, for gate state and context_length).
        """
        controller = self

        def gated_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            # ---- Fast path: prompt prefill or gate not active ---- #
            if not (controller.triggered and q_len == 1):
                return original_bound(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            # ---- Gated decode path (q_len == 1, gate active) ---- #
            # Full re-implementation of LlamaAttention.forward so we can
            # intercept attn_output between softmax and o_proj.
            # Kept identical to transformers==4.35.2 except for the gate block.

            m = original_bound.__self__   # the LlamaAttention module instance

            # -- Q / K / V projections --
            if m.config.pretraining_tp > 1:
                tp = m.config.pretraining_tp
                kv_slice = (m.num_key_value_heads * m.head_dim) // tp
                q_slices = m.q_proj.weight.split((m.num_heads * m.head_dim) // tp, dim=0)
                k_slices = m.k_proj.weight.split(kv_slice, dim=0)
                v_slices = m.v_proj.weight.split(kv_slice, dim=0)
                query_states  = torch.cat([F.linear(hidden_states, q_slices[i]) for i in range(tp)], dim=-1)
                key_states    = torch.cat([F.linear(hidden_states, k_slices[i]) for i in range(tp)], dim=-1)
                value_states  = torch.cat([F.linear(hidden_states, v_slices[i]) for i in range(tp)], dim=-1)
            else:
                query_states  = m.q_proj(hidden_states)
                key_states    = m.k_proj(hidden_states)
                value_states  = m.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, m.num_heads,           m.head_dim).transpose(1, 2)
            key_states   = key_states  .view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]

            cos, sin = m.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            if past_key_value is not None:
                key_states   = torch.cat([past_key_value[0], key_states],   dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None

            key_states   = repeat_kv(key_states,   m.num_key_value_groups)
            value_states = repeat_kv(value_states, m.num_key_value_groups)

            # -- Attention weights (pre-softmax) --
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                / math.sqrt(m.head_dim)
            )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Softmax in fp32 for numerical stability, cast back
            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            # attn_weights: (B=1, num_heads, q_len=1, kv_len)
            attn_output = torch.matmul(attn_weights, value_states)
            # attn_output: (B=1, num_heads, q_len=1, head_dim)

            # ===== FIX 1 + FIX 2: gate here, BEFORE o_proj ===== #
            ctx       = controller.context_length
            attn_row  = attn_weights[0, :, -1, :]           # (H, kv_len)
            attn_ctx  = attn_row[:, :ctx].sum(-1)            # (H,) prompt attn
            attn_new  = attn_row[:, ctx:].sum(-1)            # (H,) generated attn
            lr        = attn_ctx / (attn_ctx + attn_new + 1e-10)   # (H,) ∈ [0,1]

            mode   = controller.gate_mode
            cutoff = controller.lr_cutoff

            if mode == 'soft':
                gate = torch.sigmoid((lr - cutoff) * controller.alpha)
            elif mode == 'hard':
                gate = (lr >= cutoff).float()
            elif mode == 'zero_all':
                gate = torch.zeros_like(lr)
            elif mode == 'zero_high':
                gate = torch.ones_like(lr)
                gate[lr >= cutoff] = 0.0
            elif mode == 'zero_low':
                gate = torch.ones_like(lr)
                gate[lr < cutoff] = 0.0
            else:
                gate = torch.ones_like(lr)

            gate        = gate.to(device=attn_output.device, dtype=attn_output.dtype)
            gate        = gate.view(1, m.num_heads, 1, 1)
            attn_output = attn_output * gate
            # ==================================================== #

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, m.hidden_size)

            if m.config.pretraining_tp > 1:
                tp = m.config.pretraining_tp
                attn_output  = attn_output.split(m.hidden_size // tp, dim=2)
                o_slices     = m.o_proj.weight.split(m.hidden_size // tp, dim=1)
                attn_output  = sum(F.linear(attn_output[i], o_slices[i]) for i in range(tp))
            else:
                attn_output = m.o_proj(attn_output)

            attn_weights_ret = attn_weights if output_attentions else None
            return attn_output, attn_weights_ret, past_key_value

        return gated_forward

    # ---- Control ---------------------------------------------------- #

    def trigger(self):
        """Activate gating for the next generation call."""
        self.triggered = True

    def reset(self):
        """Deactivate gating (passthrough mode)."""
        self.triggered = False

    def remove(self):
        """Restore all original forward methods (call once after inference)."""
        for attn_mod, original in self._patched:
            attn_mod.forward = original
        self._patched.clear()
        logging.info("LookbackGatedAttention: all patches removed.")


# ------------------------------------------------------------------ #
# Accuracy helpers                                                     #
# ------------------------------------------------------------------ #

def compute_f1(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth strings."""
    pred_tokens = prediction.lower().split()
    gt_tokens   = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common   = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(x, y):
    """Length of the Longest Common Subsequence (space-optimized)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(prediction, reference):
    """ROUGE-L F1 score (token-level LCS)."""
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(prediction, answers, threshold=0.5, use_rouge=False):
    """Binary accuracy: best-match metric >= threshold → 1.0 else 0.0.

    QA:            token-F1 >= 0.5   (SQuAD-style)
    Summarization: ROUGE-L  >= 0.2   (use_rouge=True)
    """
    if not answers:
        return 0.0
    if use_rouge:
        best = max(compute_rouge_l(prediction, ans) for ans in answers)
    else:
        best = max(compute_f1(prediction, ans) for ans in answers)
    return 1.0 if best >= threshold else 0.0


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="SEP-Triggered Lookback Gating Inference (v2)"
    )
    parser.add_argument(
        "--dataset", required=True, choices=ALL_DATASETS,
        help="Dataset to evaluate. Use 'xsum' for the Lookback Gate experiment."
    )
    parser.add_argument(
        "--token_type", choices=["TBG", "SLT"], default="TBG",
        help="Which SEP probe to use (default: TBG)"
    )
    parser.add_argument(
        "--sep_threshold", type=float, default=0.5,
        help="SEP uncertainty score above which gating is triggered (default: 0.5)"
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0,
        help="Gate sharpness: gate = sigmoid((LR - cutoff) * alpha). "
             "Higher → harder soft-gate (default: 10.0)"
    )
    parser.add_argument(
        "--lr_cutoff", type=float, default=0.5,
        help="Lookback Ratio cutoff for the gate.  Heads with LR < cutoff "
             "are suppressed (default: 0.5)"
    )
    parser.add_argument(
        "--gate_mode", choices=["soft", "hard"], default="soft",
        help="'soft': σ((LR−cutoff)×α), 'hard': (LR≥cutoff)?1:0  (default: soft)"
    )
    parser.add_argument(
        "--layer_range", type=str, default=None,
        help="Comma-separated start,end for gated layers, e.g. '21,32'. "
             "Default: upper third of all layers."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override max_new_tokens. Default: 100 for xsum, 50 for QA datasets."
    )
    parser.add_argument(
        "--acc_threshold", type=float, default=None,
        help="Metric threshold for 'correct'. Default: 0.5 for QA, 0.2 for xsum."
    )
    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    dataset = args.dataset

    # Dataset-dependent defaults
    is_xsum        = (dataset in XSUM_DATASETS)
    max_new_tokens = args.max_new_tokens or (XSUM_MAX_NEW_TOKENS if is_xsum else MAX_NEW_TOKENS)
    acc_threshold  = args.acc_threshold  or (XSUM_ACC_THRESHOLD  if is_xsum else 0.5)
    use_rouge      = is_xsum
    metric_name    = "ROUGE-L" if use_rouge else "Token-F1"

    logging.info(
        f"Dataset: {dataset}  |  max_new_tokens: {max_new_tokens}  "
        f"|  metric: {metric_name} >= {acc_threshold}  "
        f"|  gate_mode: {args.gate_mode}  |  lr_cutoff: {args.lr_cutoff}"
    )

    # ---- Paths ----
    out_dir     = os.path.join(OUTPUT_BASE, dataset)
    gen_file    = os.path.join(out_dir, "generations.pkl")
    probe_file  = os.path.join(out_dir, f"sep_probe_{args.token_type}.pkl")
    result_file = os.path.join(out_dir, "gated_results.pkl")

    for path, desc in [(gen_file,   "generations.pkl"),
                       (probe_file, f"sep_probe_{args.token_type}.pkl")]:
        if not os.path.exists(path):
            logging.error(
                f"{desc} not found at {path}. "
                f"Run run_qa_generation.py and train_probe.py --save_probe first."
            )
            return

    # ---- Load ----
    logging.info(f"Loading generations from {gen_file} ...")
    with open(gen_file, "rb") as f:
        gen_data = pickle.load(f)

    logging.info(f"Loading SEP probe from {probe_file} ...")
    with open(probe_file, "rb") as f:
        probe_bundle = pickle.load(f)

    clf       = probe_bundle['clf']
    r_start   = probe_bundle['r_start']
    r_end     = probe_bundle['r_end']
    se_thresh = probe_bundle['threshold']

    logging.info(f"Probe: layers [{r_start},{r_end}), SE threshold={se_thresh:.4f}")
    logging.info(f"Gate trigger: SEP uncertainty > {args.sep_threshold}")

    # ---- Parse custom layer range ----
    layer_range = None
    if args.layer_range is not None:
        s, e = args.layer_range.split(",")
        layer_range = range(int(s), int(e))

    # ---- Load LLM ----
    logging.info(f"Loading model: {MODEL_NAME} ...")
    hf_model = HuggingfaceModel(
        model_name=MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=max_new_tokens,
    )
    raw_model = hf_model.model
    tokenizer = hf_model.tokenizer
    stop_seqs = hf_model.stop_sequences

    # For XSum, relax stopping criteria: don't stop at single newlines
    # inside the summary — only stop at double-newline or EOS.
    if is_xsum:
        stop_seqs = ['\n\n', tokenizer.eos_token]

    # ---- Score all samples with SEP probe ----
    logging.info("Scoring all samples with SEP probe ...")
    sep_scores  = []
    valid_items = []

    for item in gen_data:
        emb = item.get('tbg_embedding' if args.token_type == "TBG" else 'slt_embedding')
        if emb is None:
            sep_scores.append(None)
            valid_items.append(False)
            continue

        emb_sq  = emb.squeeze(1) if emb.dim() == 3 else emb
        feature = np.concatenate(
            [emb_sq[l].numpy() for l in range(r_start, r_end)], axis=0
        )[np.newaxis, :]

        score = clf.predict_proba(feature)[0, 1]
        sep_scores.append(score)
        valid_items.append(True)

    n_triggered   = sum(1 for s in sep_scores if s is not None and s > args.sep_threshold)
    n_passthrough = sum(1 for s, v in zip(sep_scores, valid_items)
                        if v and s <= args.sep_threshold)
    n_invalid     = sum(1 for v in valid_items if not v)

    logging.info(
        f"SEP scored {len(gen_data)} samples: "
        f"triggered={n_triggered}, passthrough={n_passthrough}, "
        f"no-embedding={n_invalid}"
    )

    # ---- Install monkey-patched gated attention ----
    # (no-ops until controller.trigger() is called)
    controller = LookbackGatedAttention(
        model=raw_model,
        context_length=0,
        alpha=args.alpha,
        lr_cutoff=args.lr_cutoff,
        gate_mode=args.gate_mode,
        layer_range=layer_range,
    )

    # ---- Inference loop ----
    results  = []
    acc_gate = []
    acc_pass = []

    logging.info("Running gated inference ...")
    for i, item in enumerate(tqdm(gen_data)):
        score    = sep_scores[i]
        answers  = item.get('answers', [])
        existing = item.get('most_likely_answer', "")
        prompt   = item.get('prompt_used', "")

        if score is None or prompt == "":
            results.append({
                **item,
                'gated_answer':   existing,
                'sep_score':      score,
                'gate_triggered': False,
            })
            acc_pass.append(compute_accuracy(existing, answers, acc_threshold, use_rouge))
            continue

        triggered = score > args.sep_threshold

        if not triggered:
            results.append({
                **item,
                'gated_answer':   existing,
                'sep_score':      float(score),
                'gate_triggered': False,
            })
            acc_pass.append(compute_accuracy(existing, answers, acc_threshold, use_rouge))

        else:
            # Triggered: re-generate with Lookback Gate active.
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            n_prompt_tokens = inputs['input_ids'].shape[1]

            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=stop_seqs,
                initial_length=n_prompt_tokens,
                tokenizer=tokenizer,
            )])

            controller.context_length = n_prompt_tokens
            controller.trigger()

            try:
                with torch.no_grad():
                    out = raw_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        # output_attentions is NOT needed: the monkey-patched
                        # forward computes attention weights internally.
                        output_attentions=False,
                        output_scores=False,
                        output_hidden_states=False,
                        return_dict_in_generate=True,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_tokens = out.sequences[0][n_prompt_tokens:]
                gated_ans  = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                for stop in stop_seqs:
                    if gated_ans.endswith(stop):
                        gated_ans = gated_ans[:-len(stop)].strip()
                        break

            except Exception as e:
                logging.error(f"Sample {i}: gated generation failed — {e}")
                gated_ans = existing

            finally:
                controller.reset()

            results.append({
                **item,
                'gated_answer':   gated_ans,
                'sep_score':      float(score),
                'gate_triggered': True,
            })
            acc_gate.append(compute_accuracy(gated_ans, answers, acc_threshold, use_rouge))

            if (i + 1) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # ---- Restore original forwards ----
    controller.remove()

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"GATED INFERENCE RESULTS — {dataset}")
    print(f"{'='*65}")
    print(f"Model:             {MODEL_NAME}")
    print(f"SEP probe:         {args.token_type}  (layers [{r_start},{r_end}))")
    print(f"Gate trigger:      SEP score > {args.sep_threshold}")
    if args.gate_mode == 'soft':
        print(f"Gate formula:      sigmoid((LR - {args.lr_cutoff}) * {args.alpha})")
    else:
        print(f"Gate formula:      hard (LR >= {args.lr_cutoff})")
    print(f"max_new_tokens:    {max_new_tokens}")
    print(f"Accuracy metric:   {metric_name} >= {acc_threshold:.2f}")
    print()
    print(f"Total samples:       {len(results)}")
    print(f"  Triggered (gated): {n_triggered}")
    print(f"  Passthrough:       {n_passthrough}")
    print(f"  No embedding:      {n_invalid}")
    print()

    if acc_gate:
        orig_on_triggered = np.mean([
            compute_accuracy(r['most_likely_answer'], r['answers'], acc_threshold, use_rouge)
            for r in results if r['gate_triggered']
        ])
        gate_mean = np.mean(acc_gate)
        delta     = gate_mean - orig_on_triggered
        print(f"Accuracy on triggered samples (N={len(acc_gate)}):")
        print(f"  Before gating (original): {orig_on_triggered:.4f}")
        print(f"  After  gating (new):      {gate_mean:.4f}  "
              f"({'↑ +' if delta >= 0 else '↓ '}{abs(delta):.4f})")

    if acc_pass:
        print(f"Accuracy passthrough (N={len(acc_pass)}): {np.mean(acc_pass):.4f}")

    all_gated = [compute_accuracy(r['gated_answer'],       r['answers'], acc_threshold, use_rouge)
                 for r in results]
    all_orig  = [compute_accuracy(r['most_likely_answer'], r['answers'], acc_threshold, use_rouge)
                 for r in results if r.get('most_likely_answer', '') != '']
    print()
    print(f"Overall accuracy (original answers): {np.mean(all_orig):.4f}")
    print(f"Overall accuracy (after gating):     {np.mean(all_gated):.4f}")
    print("=" * 65)

    # ---- Save ----
    with open(result_file, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Saved {len(results)} results → {result_file}")


if __name__ == "__main__":
    main()
