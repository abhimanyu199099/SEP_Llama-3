"""Extract Lookback Ratio features (Chuang et al., EMNLP 2024 – Lookback Lens).

For each sample in generations.pkl, re-runs the low-temperature forward pass
with output_attentions=True to compute:

  LookbackRatio[l, h] = mean over T generated tokens of:
      (sum of attention weights to context/prompt tokens)
      / (sum of all attention weights)

This is done per attention head (layer l, head h).

Final feature per sample: Tensor of shape (num_layers * num_heads,) — flattened.

Output saved to: output/{dataset}/lookback_features.pt
  {
    "X_lookback": Tensor  (N, num_layers * num_heads),
    "entropy":    Tensor  (N,),                           # SE score from nli_labels.json
    "sample_indices": list[int]
  }

Usage:
    python extract_lookback_features.py --dataset squad
    python extract_lookback_features.py --dataset trivia_qa
"""
import os
import sys
import gc
import json
import pickle
import logging
import argparse

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "semantic_uncertainty"))

from uncertainty.models.huggingface_models import HuggingfaceModel
from common_utils import (
    MODEL_NAME, QA_DATASETS, OUTPUT_BASE,
    TEMPERATURE_LOW, MAX_NEW_TOKENS,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Lookback Ratio features")
    parser.add_argument("--dataset", required=True, choices=QA_DATASETS,
                        help="QA dataset name")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Samples to process before flushing GPU cache (default: 1)")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save intermediate checkpoint every N samples")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir   = os.path.join(OUTPUT_BASE, args.dataset)
    gen_file    = os.path.join(input_dir, "generations.pkl")
    nli_file    = os.path.join(input_dir, "nli_labels.json")
    output_file = os.path.join(input_dir, "lookback_features.pt")

    # ------------------------------------------------------------------ #
    # 1.  Load previously generated data                                  #
    # ------------------------------------------------------------------ #
    logging.info(f"Loading generations from {gen_file} ...")
    with open(gen_file, "rb") as f:
        gen_data = pickle.load(f)

    logging.info(f"Loading NLI labels from {nli_file} ...")
    with open(nli_file, "r") as f:
        nli_data = json.load(f)

    # Index nli data by sample_index
    nli_by_idx = {item['sample_index']: item for item in nli_data}
    # Only process samples that have NLI labels (same filter as extract_all_layers.py)
    valid_gen = [item for item in gen_data if item['sample_index'] in nli_by_idx]
    logging.info(f"{len(valid_gen)} samples have both generations and NLI labels.")

    # ------------------------------------------------------------------ #
    # 2.  Load LLM                                                        #
    # ------------------------------------------------------------------ #
    logging.info(f"Loading model: {MODEL_NAME} ...")
    model = HuggingfaceModel(
        model_name=MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ------------------------------------------------------------------ #
    # 3.  Re-run low-temp forward pass with output_attentions=True        #
    # ------------------------------------------------------------------ #
    X_lookback_list   = []
    entropy_list      = []
    sample_index_list = []
    skipped           = 0

    logging.info("Extracting lookback ratio features ...")
    for i, gen_item in enumerate(tqdm(valid_gen)):
        sample_idx = gen_item['sample_index']
        prompt     = gen_item['prompt_used']

        try:
            _, _, _, lookback_ratio = model.predict(
                prompt,
                temperature=TEMPERATURE_LOW,
                return_latent=False,
                return_attention=True,
            )
            # lookback_ratio: (num_layers, num_heads) or None
            if lookback_ratio is None:
                logging.warning(f"Sample {sample_idx}: lookback_ratio is None, skipping.")
                skipped += 1
                continue

            # Flatten to (num_layers * num_heads,)
            X_lookback_list.append(lookback_ratio.flatten().cpu())
            entropy_list.append(float(nli_by_idx[sample_idx]['entropy']))
            sample_index_list.append(sample_idx)

        except Exception as e:
            logging.error(f"Sample {sample_idx}: failed — {e}")
            skipped += 1

        # Periodic GPU + memory cleanup
        if (i + 1) % args.batch_size == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Periodic checkpoint
        if len(X_lookback_list) > 0 and (i + 1) % args.checkpoint_every == 0:
            X_lb_ckpt   = torch.stack(X_lookback_list)
            ent_ckpt    = torch.tensor(entropy_list, dtype=torch.float32)
            torch.save({
                "X_lookback":     X_lb_ckpt,
                "entropy":        ent_ckpt,
                "sample_indices": sample_index_list,
            }, output_file + ".ckpt")
            logging.info(f"Checkpoint at {i+1}/{len(valid_gen)} — "
                         f"feature shape: {X_lb_ckpt.shape}, skipped: {skipped}")

    # ------------------------------------------------------------------ #
    # 4.  Assemble and save                                               #
    # ------------------------------------------------------------------ #
    if len(X_lookback_list) == 0:
        logging.error("No valid samples found. Nothing saved.")
        return

    X_lookback = torch.stack(X_lookback_list)       # (N, num_layers * num_heads)
    entropy    = torch.tensor(entropy_list, dtype=torch.float32)   # (N,)

    num_layers_x_heads = X_lookback.shape[1]
    logging.info(f"Lookback features: {X_lookback.shape}  "
                 f"(N={X_lookback.shape[0]}, "
                 f"num_layers*num_heads={num_layers_x_heads})")
    logging.info(f"Skipped {skipped} samples.")
    logging.info(f"Entropy stats: min={entropy.min():.4f}  max={entropy.max():.4f}  "
                 f"mean={entropy.mean():.4f}")

    torch.save({
        "X_lookback":     X_lookback,
        "entropy":        entropy,
        "sample_indices": sample_index_list,
    }, output_file)
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
