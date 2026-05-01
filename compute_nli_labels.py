"""Compute NLI-based semantic entropy labels for QA generations.

Supports --dataset for QA datasets with condition_on_question and
strict_entailment matching OATML paper defaults.

Usage:
    python compute_nli_labels.py --dataset squad
    python compute_nli_labels.py --dataset trivia_qa --no-strict_entailment
"""
import torch
import pickle
import json
import argparse
import numpy as np
import logging
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common_utils import (
    NLI_MODEL, QA_DATASETS, ALL_DATASETS, RAGTRUTH_DATASETS,
    OUTPUT_BASE, CONDITION_ON_QUESTION, STRICT_ENTAILMENT,
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="NLI-based Semantic Entropy Labels")
    _nli_datasets = [d for d in ALL_DATASETS if d not in RAGTRUTH_DATASETS]
    parser.add_argument("--dataset", required=True, choices=_nli_datasets,
                        help="Dataset name (all except ragtruth, which uses map_ragtruth_labels.py)")
    parser.add_argument("--condition_on_question", action=argparse.BooleanOptionalAction,
                        default=CONDITION_ON_QUESTION,
                        help="Prepend question to generations for NLI")
    parser.add_argument("--strict_entailment", action=argparse.BooleanOptionalAction,
                        default=STRICT_ENTAILMENT,
                        help="Require both NLI directions to be entailment (class 2)")
    return parser.parse_args()


def check_implication(model, tokenizer, premise, hypothesis):
    """Returns NLI class: 0=Contradiction, 1=Neutral, 2=Entailment."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model(**inputs)
    return torch.argmax(output.logits, dim=1).item()


def are_equivalent(model, tokenizer, text1, text2, strict=True):
    """
    Bidirectional entailment check matching OATML semantic_entropy.py:get_semantic_ids.

    strict=True  (OATML default): both directions must be entailment (class 2)
    strict=False: no contradiction AND not both neutral
    """
    impl1 = check_implication(model, tokenizer, text1, text2)
    impl2 = check_implication(model, tokenizer, text2, text1)

    if strict:
        return (impl1 == 2) and (impl2 == 2)
    else:
        implications = [impl1, impl2]
        return (0 not in implications) and ([1, 1] != implications)


def get_semantic_ids(generations, model, tokenizer, strict=True):
    """
    Group generations into semantic clusters.
    Returns list of cluster IDs, one per generation.
    """
    semantic_set_ids = [-1] * len(generations)
    next_id = 0
    for i in range(len(generations)):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(generations)):
                if semantic_set_ids[j] == -1:
                    if are_equivalent(model, tokenizer,
                                      generations[i], generations[j], strict=strict):
                        semantic_set_ids[j] = next_id
            next_id += 1
    assert -1 not in semantic_set_ids
    return semantic_set_ids


def cluster_assignment_entropy(semantic_ids):
    """OATML cluster_assignment_entropy: -sum(p * ln(p)) using natural log."""
    counts = np.bincount(semantic_ids)
    probabilities = counts / len(semantic_ids)
    return -np.sum(probabilities * np.log(probabilities))


def main():
    args = parse_args()

    input_dir = os.path.join(OUTPUT_BASE, args.dataset)
    input_file = os.path.join(input_dir, "generations.pkl")
    output_file = os.path.join(input_dir, "nli_labels.json")

    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found. Run run_qa_generation.py first.")
        return

    logging.info(f"Loading NLI Model: {NLI_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to("cuda")

    logging.info(f"Loading generations from {input_file}...")
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    logging.info(f"Settings: condition_on_question={args.condition_on_question}, "
                 f"strict_entailment={args.strict_entailment}")

    output_data = []
    skipped = 0

    logging.info(f"Computing Semantic Entropy via NLI for {len(data)} items...")

    for item in tqdm(data):
        gens = [g for g in item['generations'] if g.strip()]
        if len(gens) < 2:
            skipped += 1
            continue

        # Condition on question: prepend question to each generation for NLI
        question = item.get('question', '')
        if args.condition_on_question and question:
            gens_for_nli = [f"{question} {g}" for g in gens]
        else:
            gens_for_nli = gens

        semantic_ids = get_semantic_ids(
            gens_for_nli, model, tokenizer, strict=args.strict_entailment
        )
        entropy = cluster_assignment_entropy(semantic_ids)

        output_data.append({
            "sample_index": item['sample_index'],
            "question": question,
            "entropy": float(entropy),
            "num_clusters": max(semantic_ids) + 1,
        })

    logging.info(f"Done. Retained {len(output_data)} samples, skipped {skipped} (empty generations).")
    if output_data:
        logging.info(f"Entropy stats: min={min(d['entropy'] for d in output_data):.4f}, "
                     f"max={max(d['entropy'] for d in output_data):.4f}, "
                     f"mean={np.mean([d['entropy'] for d in output_data]):.4f}")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
