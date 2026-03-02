"""Causal Validation of SEP-Triggered Lookback Gating (v2)

Two tests adapted from the project plan (Yu et al. + Chuang et al.):

Test 1 – Knockout Test
    Hypothesis: Heads flagged as "hallucinating" by the gate actually CAUSE the
    degraded output. If we zero them out completely (gate=0, harder than the
    sigmoid gate), accuracy should drop further or stay the same, not improve.

    More importantly: gated re-generations should be BETTER than the original
    answers on triggered samples. The knockout comparison exposes the delta.

Test 2 – Blindness Test
    Hypothesis: Heads with HIGH Lookback Ratio (strongly attending to the
    prompt/context) are the "grounding" heads. Suppressing them (forcing gate=0
    for *high-LR* heads instead of low-LR heads) should DEGRADE output quality,
    demonstrating these heads are causally responsible for correct grounding.

v2 changes (matching inference_with_gate.py v2):
  - ForcedGateController replaced with MonkeyPatchForcedController using
    per-instance monkey-patching of LlamaAttention.forward (fixes off-by-one).
  - Gate math consistent: LR values compared at 0.5 cutoff.
  - Dataset choices extended to include 'xsum', accuracy threshold is dataset-aware.

Inputs expected in output/{dataset}/:
    - gated_results.pkl      (from inference_with_gate.py)
    - sep_probe_{token}.pkl  (from train_probe.py --save_probe)

Usage:
    python causal_validation.py --dataset xsum
    python causal_validation.py --dataset squad --num_samples 50
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
# Import shared metric helpers from inference_with_gate to avoid duplication.
# compute_accuracy handles both Token-F1 (QA) and ROUGE-L (XSum) via use_rouge flag.
from inference_with_gate import compute_accuracy  # noqa: E402
from common_utils import (
    MODEL_NAME, ALL_DATASETS, XSUM_DATASETS, OUTPUT_BASE,
    MAX_NEW_TOKENS, XSUM_MAX_NEW_TOKENS, XSUM_ACC_THRESHOLD,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


# ------------------------------------------------------------------ #
# MonkeyPatchForcedController                                         #
#                                                                     #
# Same architecture as MonkeyPatchGateController in inference_with_  #
# gate.py but applies FORCED (binary) gate strategies instead of the #
# soft sigmoid, for causal ablation tests.                           #
# ------------------------------------------------------------------ #

class MonkeyPatchForcedController:
    """Per-instance monkey-patch of LlamaAttention.forward for causal ablation.

    Like MonkeyPatchGateController except it uses a forced gate strategy
    rather than the learned sigmoid:

      mode='zero_all'   : gate = 0 for ALL heads in target layers (knockout).
      mode='zero_high'  : gate = 0 for heads with LR ≥ lr_cutoff (blindness).
      mode='zero_low'   : gate = 0 for heads with LR < lr_cutoff  (hard control).

    This produces a clean binary ablation that tests causality directly,
    without the softness of the sigmoid gate obscuring the result.
    """

    def __init__(self, model, context_length, layer_range=None,
                 mode='zero_all', lr_cutoff=0.5):
        self.triggered      = False
        self.context_length = context_length
        self.mode           = mode
        self.lr_cutoff      = lr_cutoff
        self._patched       = []   # list of (attn_mod, original_bound_method)

        layers = model.model.layers
        n = len(layers)
        if layer_range is None:
            layer_range = range(n * 2 // 3, n)

        for idx in layer_range:
            attn_mod = layers[idx].self_attn
            original = attn_mod.forward
            attn_mod.forward = self._make_forced_forward(original)
            self._patched.append((attn_mod, original))

        logging.info(
            f"MonkeyPatchForcedController (mode={mode}): patched "
            f"{len(self._patched)} layers, lr_cutoff={lr_cutoff}"
        )

    # ---------------------------------------------------------------- #
    # Patched forward factory                                           #
    # ---------------------------------------------------------------- #

    def _make_forced_forward(self, original_bound):
        controller = self

        def forced_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            # Fast path: prompt prefill or controller not triggered
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

            # ---- Gated decode path (q_len == 1, controller active) ---- #

            m = original_bound.__self__

            # Q / K / V projections
            if m.config.pretraining_tp > 1:
                tp = m.config.pretraining_tp
                kv_slice = (m.num_key_value_heads * m.head_dim) // tp
                q_slices = m.q_proj.weight.split((m.num_heads * m.head_dim) // tp, dim=0)
                k_slices = m.k_proj.weight.split(kv_slice, dim=0)
                v_slices = m.v_proj.weight.split(kv_slice, dim=0)
                query_states = torch.cat([F.linear(hidden_states, q_slices[i]) for i in range(tp)], dim=-1)
                key_states   = torch.cat([F.linear(hidden_states, k_slices[i]) for i in range(tp)], dim=-1)
                value_states = torch.cat([F.linear(hidden_states, v_slices[i]) for i in range(tp)], dim=-1)
            else:
                query_states = m.q_proj(hidden_states)
                key_states   = m.k_proj(hidden_states)
                value_states = m.v_proj(hidden_states)

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

            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                / math.sqrt(m.head_dim)
            )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            dropout_p = getattr(m, 'attention_dropout', 0.0)
            if dropout_p > 0.0 and m.training:
                attn_weights = F.dropout(attn_weights, p=dropout_p)

            attn_output = torch.matmul(attn_weights, value_states)
            # attn_output: (B=1, num_heads, q_len=1, head_dim)

            # ---- Forced gate ---- #
            ctx      = controller.context_length
            attn_row = attn_weights[0, :, -1, :]           # (H, kv_len)
            attn_ctx = attn_row[:, :ctx].sum(-1)            # (H,)
            attn_new = attn_row[:, ctx:].sum(-1)            # (H,)
            lr       = attn_ctx / (attn_ctx + attn_new + 1e-10)  # (H,) ∈ [0,1]

            gate = torch.ones_like(lr)     # default: pass-through

            if controller.mode == 'zero_all':
                gate = torch.zeros_like(lr)
            elif controller.mode == 'zero_high':
                gate[lr >= controller.lr_cutoff] = 0.0
            elif controller.mode == 'zero_low':
                gate[lr < controller.lr_cutoff] = 0.0
            # ------------------- #

            gate        = gate.to(attn_output.device).view(1, m.num_heads, 1, 1)
            attn_output = attn_output * gate

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, m.hidden_size)

            if m.config.pretraining_tp > 1:
                tp = m.config.pretraining_tp
                attn_output = attn_output.split(m.hidden_size // tp, dim=2)
                o_slices    = m.o_proj.weight.split(m.hidden_size // tp, dim=1)
                attn_output = sum(F.linear(attn_output[i], o_slices[i]) for i in range(tp))
            else:
                attn_output = m.o_proj(attn_output)

            attn_weights_ret = attn_weights if output_attentions else None
            return attn_output, attn_weights_ret, past_key_value

        return forced_forward

    # ---------------------------------------------------------------- #
    # Control                                                            #
    # ---------------------------------------------------------------- #

    def trigger(self):
        self.triggered = True

    def reset(self):
        self.triggered = False

    def remove(self):
        for attn_mod, original in self._patched:
            attn_mod.forward = original
        self._patched.clear()
        logging.info("MonkeyPatchForcedController: all patches removed.")


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def compute_f1(pred, gt):
    p, g = pred.lower().split(), gt.lower().split()
    if not p or not g:
        return 0.0
    common   = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec  = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def generate_with_controller(raw_model, tokenizer, prompt,
                              controller, stop_seqs, max_new_tokens,
                              existing_fallback):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    n_prompt = inputs['input_ids'].shape[1]
    controller.context_length = n_prompt
    controller.trigger()

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        stops=stop_seqs,
        initial_length=n_prompt,
        tokenizer=tokenizer,
    )])

    try:
        with torch.no_grad():
            out = raw_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                output_attentions=False,       # not needed: monkey-patch computes internally
                output_scores=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tok = out.sequences[0][n_prompt:]
        ans = tokenizer.decode(gen_tok, skip_special_tokens=True).strip()
        for stop in stop_seqs:
            if ans.endswith(stop):
                ans = ans[:-len(stop)].strip()
                break
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        ans = existing_fallback
    finally:
        controller.reset()

    return ans


# ------------------------------------------------------------------ #
# Parse args                                                          #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal Validation of Lookback Gate (v2)"
    )
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--token_type", choices=["TBG", "SLT"], default="TBG")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Samples to test per condition (default: 100)")
    parser.add_argument("--lr_cutoff", type=float, default=0.5,
                        help="LR threshold for zero_high/zero_low modes (default: 0.5)")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="Comma-separated start,end, e.g. '21,32'")
    parser.add_argument("--acc_threshold", type=float, default=None,
                        help="F1 threshold for 'correct'. Default: 0.5 QA, 0.2 xsum.")
    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    ds = args.dataset

    is_xsum        = (ds in XSUM_DATASETS)
    use_rouge      = is_xsum
    max_new_tokens = XSUM_MAX_NEW_TOKENS if is_xsum else MAX_NEW_TOKENS
    acc_threshold  = args.acc_threshold or (XSUM_ACC_THRESHOLD if is_xsum else 0.5)
    metric_name    = "ROUGE-L" if use_rouge else "Token-F1"

    out_dir      = os.path.join(OUTPUT_BASE, ds)
    gated_file   = os.path.join(out_dir, "gated_results.pkl")
    probe_file   = os.path.join(out_dir, f"sep_probe_{args.token_type}.pkl")
    val_out_file = os.path.join(out_dir, "causal_validation.pkl")

    for path, desc in [(gated_file, "gated_results.pkl"),
                       (probe_file, f"sep_probe_{args.token_type}.pkl")]:
        if not os.path.exists(path):
            logging.error(f"{desc} not found at {path}.")
            return

    with open(gated_file, "rb") as f:
        gated_data = pickle.load(f)

    layer_range = None
    if args.layer_range:
        s, e = args.layer_range.split(",")
        layer_range = range(int(s), int(e))

    # ================================================================ #
    # Sample selection                                                  #
    # Test 1 (knockout): samples that WERE triggered by SEP             #
    # Test 2 (blindness): samples that were NOT triggered               #
    # ================================================================ #

    triggered_items   = [r for r in gated_data if r.get('gate_triggered', False)]
    passthrough_items = [r for r in gated_data if not r.get('gate_triggered', False)
                         and r.get('prompt_used', '') != '']

    n = min(args.num_samples, len(triggered_items), len(passthrough_items))
    if n == 0:
        logging.error("Not enough samples for validation. "
                      "Run inference_with_gate.py first.")
        return

    knock_items = triggered_items[:n]
    blind_items = passthrough_items[:n]

    logging.info(f"Accuracy threshold: {metric_name} >= {acc_threshold}")
    logging.info(f"Test 1 (knockout):  {len(knock_items)} triggered samples")
    logging.info(f"Test 2 (blindness): {len(blind_items)} passthrough samples")

    # ---- Load LLM ----
    logging.info(f"Loading model: {MODEL_NAME} ...")
    hf_model  = HuggingfaceModel(
        model_name=MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=max_new_tokens,
    )
    raw_model = hf_model.model
    tokenizer = hf_model.tokenizer
    stop_seqs = hf_model.stop_sequences

    # For XSum, relax stopping criteria: don't stop at single newlines.
    if ds in XSUM_DATASETS:
        stop_seqs = ['\n\n', tokenizer.eos_token]

    # ================================================================ #
    # TEST 1: KNOCKOUT (zero_all)                                       #
    # ================================================================ #

    logging.info("\n=== TEST 1: KNOCKOUT (zero all upper-layer heads) ===")
    knockout_ctrl = MonkeyPatchForcedController(
        raw_model, context_length=0,
        layer_range=layer_range, mode='zero_all'
    )

    knock_results = []
    for item in tqdm(knock_items, desc="Knockout"):
        prompt  = item['prompt_used']
        answers = item.get('answers', [])
        orig    = item['most_likely_answer']
        gated   = item['gated_answer']

        ko_ans = generate_with_controller(
            raw_model, tokenizer, prompt,
            knockout_ctrl, stop_seqs, max_new_tokens, orig
        )
        knock_results.append({
            'question':        item.get('question', ''),
            'answers':         answers,
            'original_answer': orig,
            'gated_answer':    gated,
            'knockout_answer': ko_ans,
            'acc_original':    compute_accuracy(orig,   answers, acc_threshold, use_rouge),
            'acc_gated':       compute_accuracy(gated,  answers, acc_threshold, use_rouge),
            'acc_knockout':    compute_accuracy(ko_ans, answers, acc_threshold, use_rouge),
        })
        gc.collect()
        torch.cuda.empty_cache()

    knockout_ctrl.remove()

    # ================================================================ #
    # TEST 2: BLINDNESS (zero high-LR / context-attending heads)        #
    # ================================================================ #

    logging.info("\n=== TEST 2: BLINDNESS (zero high-LR grounding heads) ===")
    blindness_ctrl = MonkeyPatchForcedController(
        raw_model, context_length=0,
        layer_range=layer_range, mode='zero_high', lr_cutoff=args.lr_cutoff
    )

    blind_results = []
    for item in tqdm(blind_items, desc="Blindness"):
        prompt  = item['prompt_used']
        answers = item.get('answers', [])
        orig    = item['most_likely_answer']

        blind_ans = generate_with_controller(
            raw_model, tokenizer, prompt,
            blindness_ctrl, stop_seqs, max_new_tokens, orig
        )
        blind_results.append({
            'question':         item.get('question', ''),
            'answers':          answers,
            'original_answer':  orig,
            'blindness_answer': blind_ans,
            'acc_original':     compute_accuracy(orig,      answers, acc_threshold, use_rouge),
            'acc_blindness':    compute_accuracy(blind_ans, answers, acc_threshold, use_rouge),
        })
        gc.collect()
        torch.cuda.empty_cache()

    blindness_ctrl.remove()

    # ================================================================ #
    # Results                                                           #
    # ================================================================ #

    print(f"\n{'='*65}")
    print(f"CAUSAL VALIDATION RESULTS — {ds}")
    print(f"{'='*65}")
    print(f"Model: {MODEL_NAME}")
    print(f"Accuracy metric: {metric_name} >= {acc_threshold:.2f}")
    print(f"LR cutoff (blindness test): {args.lr_cutoff}")
    print()

    orig_acc_ko  = np.mean([r['acc_original'] for r in knock_results])
    gated_acc_ko = np.mean([r['acc_gated']    for r in knock_results])
    ko_acc       = np.mean([r['acc_knockout'] for r in knock_results])

    print(f"--- TEST 1: Knockout (N={len(knock_results)} triggered samples) ---")
    print(f"  Original answer accuracy (pre-gate):   {orig_acc_ko:.4f}")
    print(f"  Gated answer accuracy (sigmoid gate):  {gated_acc_ko:.4f}  "
          f"({'↑ IMPROVED' if gated_acc_ko > orig_acc_ko else '↓ degraded'})")
    print(f"  Knockout accuracy (hard zero_all):     {ko_acc:.4f}  "
          f"({'↓ worse than gate' if ko_acc < gated_acc_ko else '→ similar'})")
    if gated_acc_ko > ko_acc:
        print(f"  → CAUSAL: sigmoid gate > hard zero confirms soft suppression is optimal")
    else:
        print(f"  → NOTE: hard zero matched or beat sigmoid gate — consider lower alpha")

    print()

    orig_acc_bl = np.mean([r['acc_original']  for r in blind_results])
    blind_acc   = np.mean([r['acc_blindness'] for r in blind_results])

    print(f"--- TEST 2: Blindness (N={len(blind_results)} passthrough samples) ---")
    print(f"  Original answer accuracy:            {orig_acc_bl:.4f}")
    print(f"  After blinding grounding heads:      {blind_acc:.4f}  "
          f"({'↓ DEGRADED — grounding heads are causal!' if blind_acc < orig_acc_bl else '→ no change'})")
    if blind_acc < orig_acc_bl:
        print(f"  → CAUSAL: zeroing high-LR heads hurts accuracy, confirming their role in grounding")
    else:
        print(f"  → NOTE: blindness test inconclusive — try a lower lr_cutoff")

    print()
    print("=" * 65)

    val_output = {
        'knockout_results':  knock_results,
        'blindness_results': blind_results,
        'config': vars(args),
    }
    with open(val_out_file, "wb") as f:
        pickle.dump(val_output, f)
    logging.info(f"Saved validation results → {val_out_file}")


if __name__ == "__main__":
    main()
