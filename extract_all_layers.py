"""Extract all-layer features from QA generation data.

Assembles TBG and SLT embedding tensors with NLI entropy scores
for probe training.

Usage:
    python extract_all_layers.py --dataset squad
"""
import torch
import json
import pickle
import argparse
import logging
import os
from common_utils import ALL_DATASETS, OUTPUT_BASE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract all-layer features")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                        help="Dataset name")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.join(OUTPUT_BASE, args.dataset)
    gen_file = os.path.join(input_dir, "generations.pkl")
    nli_file = os.path.join(input_dir, "nli_labels.json")
    output_file = os.path.join(input_dir, "all_layers.pt")

    # 1. Load generation data (contains saved TBG/SLT embeddings)
    logging.info(f"Loading generations from {gen_file}...")
    with open(gen_file, "rb") as f:
        gen_data = pickle.load(f)

    # 2. Load NLI labels (contains continuous entropy scores)
    logging.info(f"Loading NLI labels from {nli_file}...")
    with open(nli_file, "r") as f:
        nli_data = json.load(f)

    # Index generation data by sample_index for fast lookup
    gen_by_idx = {item['sample_index']: item for item in gen_data}

    X_tbg_list = []
    X_slt_list = []
    entropy_list = []
    skipped = 0

    logging.info(f"Assembling feature tensors for {len(nli_data)} labeled samples...")
    for nli_item in nli_data:
        idx = nli_item['sample_index']
        gen_item = gen_by_idx.get(idx)

        if gen_item is None:
            skipped += 1
            continue

        tbg = gen_item.get('tbg_embedding')
        slt = gen_item.get('slt_embedding')

        if tbg is None or slt is None:
            skipped += 1
            continue

        # tbg/slt shape: (num_layers, 1, hidden_dim) -> squeeze to (num_layers, hidden_dim)
        tbg_squeezed = tbg.squeeze(-2) if tbg.dim() == 3 else tbg
        slt_squeezed = slt.squeeze(-2) if slt.dim() == 3 else slt

        X_tbg_list.append(tbg_squeezed)
        X_slt_list.append(slt_squeezed)
        entropy_list.append(nli_item['entropy'])

    if len(X_tbg_list) == 0:
        logging.error("No valid samples found!")
        return

    X_tbg = torch.stack(X_tbg_list)   # (N, num_layers, hidden_dim)
    X_slt = torch.stack(X_slt_list)   # (N, num_layers, hidden_dim)
    entropy = torch.tensor(entropy_list, dtype=torch.float32)

    logging.info(f"TBG features: {X_tbg.shape}")
    logging.info(f"SLT features: {X_slt.shape}")
    logging.info(f"Entropy scores: {entropy.shape}")
    logging.info(f"Skipped {skipped} samples (missing embeddings)")

    torch.save({"X_tbg": X_tbg, "X_slt": X_slt, "entropy": entropy}, output_file)
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
