"""Generation for Semantic Entropy Probes.

Generates answers on QA datasets (SQuAD, TriviaQA, NQ, BioASQ) and
summaries on XSum using Llama-2-7B.

Usage:
    python run_qa_generation.py --dataset squad
    python run_qa_generation.py --dataset trivia_qa
    python run_qa_generation.py --dataset xsum
"""
import os
import sys
import gc
import torch
import random
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "semantic_uncertainty"))

from semantic_uncertainty.uncertainty.data.data_utils import load_ds
from semantic_uncertainty.uncertainty.models.huggingface_models import HuggingfaceModel
from common_utils import (
    MODEL_NAME, NUM_SAMPLES_QA, NUM_GENERATIONS_QA,
    TEMPERATURE_HIGH, TEMPERATURE_LOW, NUM_FEW_SHOT,
    MAX_NEW_TOKENS, SEED_QA, BRIEF_PROMPT, USE_CONTEXT,
    ALL_DATASETS, XSUM_DATASETS, OUTPUT_BASE,
    XSUM_MAX_NEW_TOKENS, XSUM_NUM_GENERATIONS, XSUM_NUM_FEW_SHOT,
    XSUM_NUM_SAMPLES, XSUM_BRIEF_PROMPT,
    CNN_DATASETS, CNN_NUM_SAMPLES, CNN_NUM_GENERATIONS,
    CNN_MAX_NEW_TOKENS, CNN_NUM_FEW_SHOT, CNN_BRIEF_PROMPT,
    HALUEVAL_DATASETS, HALUEVAL_NUM_SAMPLES, HALUEVAL_USE_CONTEXT,
)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generation for SEP (QA + XSum)")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                        help="Dataset to generate on (QA or summarization)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of validation samples (default: 2000 QA, 1000 XSum)")
    return parser.parse_args()


def get_answerable_indices(dataset):
    """Get indices where answers exist (len(answers['text']) > 0)."""
    answerable = []
    unanswerable = []
    for i in range(len(dataset)):
        ex = dataset[i]
        if len(ex['answers']['text']) > 0:
            answerable.append(i)
        else:
            unanswerable.append(i)
    return answerable, unanswerable


def build_few_shot_prompt(dataset, example_indices, brief, use_context=False):
    """Build raw-text few-shot prompt matching OATML format."""
    prompt = brief  # "Answer the following question as briefly as possible.\n"
    for idx in example_indices:
        example = dataset[idx]
        question = example["question"]
        answer = example["answers"]["text"][0]
        context = example.get("context", None)
        if use_context and context:
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        prompt += f"Answer: {answer}\n\n"
    return prompt


def build_test_input(question, context=None, use_context=False):
    """Build the test question portion of the prompt."""
    prompt = ""
    if use_context and context:
        prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer:"
    return prompt


def build_few_shot_prompt_xsum(dataset, example_indices, brief):
    """Build few-shot prompt for XSum summarization."""
    prompt = brief
    for idx in example_indices:
        example = dataset[idx]
        document = example["context"]
        summary = example["answers"]["text"][0]
        prompt += f"Article: {document}\n"
        prompt += f"Summary: {summary}\n\n"
    return prompt


def build_test_input_xsum(context):
    """Build the test input for XSum: Article + Summary prefix."""
    return f"Article: {context}\nSummary:"


def compute_f1(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(prediction, example):
    """SQuAD-style accuracy: F1 >= 50.0 against any reference answer."""
    answers = example['answers']['text']
    if not answers:
        return 0.0
    max_f1 = max(compute_f1(prediction, ans) for ans in answers)
    return 1.0 if (max_f1 * 100 >= 50.0) else 0.0


def main():
    args = parse_args()
    dataset_name = args.dataset
    is_xsum = dataset_name in XSUM_DATASETS
    is_cnn = dataset_name in CNN_DATASETS
    is_summ = is_xsum or is_cnn          # any summarization task
    is_halueval = dataset_name in HALUEVAL_DATASETS

    # Dataset-dependent defaults
    if is_xsum:
        num_samples    = args.num_samples or XSUM_NUM_SAMPLES
        num_few_shot   = XSUM_NUM_FEW_SHOT
        num_generations = XSUM_NUM_GENERATIONS
        max_new_tokens = XSUM_MAX_NEW_TOKENS
        brief_prompt   = XSUM_BRIEF_PROMPT
        use_context    = True
    elif is_cnn:
        num_samples    = args.num_samples or CNN_NUM_SAMPLES
        num_few_shot   = CNN_NUM_FEW_SHOT
        num_generations = CNN_NUM_GENERATIONS
        max_new_tokens = CNN_MAX_NEW_TOKENS
        brief_prompt   = CNN_BRIEF_PROMPT
        use_context    = True
    else:
        num_samples    = args.num_samples or (HALUEVAL_NUM_SAMPLES if is_halueval else NUM_SAMPLES_QA)
        num_few_shot   = NUM_FEW_SHOT
        num_generations = NUM_GENERATIONS_QA
        max_new_tokens = MAX_NEW_TOKENS
        brief_prompt   = BRIEF_PROMPT
        use_context    = HALUEVAL_USE_CONTEXT if is_halueval else USE_CONTEXT

    # Reproducibility
    random.seed(SEED_QA)
    np.random.seed(SEED_QA)
    torch.manual_seed(SEED_QA)

    # Output directory
    output_dir = os.path.join(OUTPUT_BASE, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generations.pkl")

    # Load dataset
    logging.info(f"Loading dataset: {dataset_name}...")
    train_ds, val_ds = load_ds(dataset_name, seed=SEED_QA)

    logging.info(f"Train set size: {len(train_ds)}")
    logging.info(f"Validation set size: {len(val_ds)}")

    # For SQuAD: filter to answerable-only
    if dataset_name == "squad":
        logging.info("SQuAD: filtering to answerable-only questions...")
        answerable_train, _ = get_answerable_indices(train_ds)
        answerable_val, _ = get_answerable_indices(val_ds)
        # For validation, create filtered list
        val_ds_filtered = [val_ds[i] for i in answerable_val]
        val_ds = val_ds_filtered
        logging.info(f"SQuAD answerable: train={len(answerable_train)}, val={len(val_ds)}")
    else:
        answerable_train, _ = get_answerable_indices(train_ds)

    # Select few-shot examples from answerable training set
    prompt_indices = random.sample(answerable_train, min(num_few_shot, len(answerable_train)))
    if is_summ:
        few_shot_prompt = build_few_shot_prompt_xsum(train_ds, prompt_indices, brief_prompt)
    else:
        few_shot_prompt = build_few_shot_prompt(train_ds, prompt_indices, brief_prompt, use_context)

    logging.info(f"Few-shot prompt ({len(prompt_indices)} examples):")
    logging.info(few_shot_prompt[:500] + "...")

    # Sample validation indices
    num_available = len(val_ds)
    actual_samples = min(num_samples, num_available)
    if num_samples > num_available:
        logging.warning(f"Requested {num_samples} but only {num_available} available. Using all.")
    indices = random.sample(range(num_available), actual_samples)
    logging.info(f"Sampled {actual_samples} validation indices.")

    # Initialize model
    logging.info(f"Loading model: {MODEL_NAME}...")
    model = HuggingfaceModel(
        model_name=MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=max_new_tokens,
    )

    # Generation loop
    data_store = []
    accuracies = []
    logging.info(f"Starting generation: {actual_samples} samples x "
                 f"{num_generations} high-temp + 1 low-temp...")

    for i, idx in tqdm(enumerate(indices), total=actual_samples):
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        example = val_ds[idx]
        question = example["question"]
        context = example.get("context", None)
        example_id = example.get("id", str(idx))

        if is_summ:
            test_input = build_test_input_xsum(context)
        else:
            test_input = build_test_input(question, context, use_context)
        local_prompt = few_shot_prompt + test_input

        # --- High-temp generations for SE computation ---
        generations = []
        for gen_i in range(num_generations):
            try:
                answer, _, _, _ = model.predict(local_prompt, temperature=TEMPERATURE_HIGH)
                generations.append(answer)
            except Exception as e:
                logging.error(f"Generation failed for sample {i}, gen {gen_i}: {e}")
                generations.append("")

        # --- Low-temp "most likely" answer with return_latent=True ---
        tbg_embedding = None
        slt_embedding = None
        ml_answer = ""
        try:
            ml_answer, _, hidden_states, _ = model.predict(
                local_prompt, temperature=TEMPERATURE_LOW, return_latent=True
            )
            # hidden_states = (last_tok_emb, slt_all_layers, tbg_all_layers)
            _, slt_embedding, tbg_embedding = hidden_states
        except Exception as e:
            logging.error(f"Latent extraction failed for sample {i}: {e}")

        # Compute accuracy on low-temp answer
        acc = compute_accuracy(ml_answer, example)
        accuracies.append(acc)

        if i == 0 or (i + 1) % 50 == 0:
            logging.info(f"Sample {i}: Q='{question[:80]}...' | "
                         f"ML='{ml_answer[:60]}' | acc={acc}")

        data_store.append({
            "sample_index": i,
            "dataset_index": idx,
            "id": example_id,
            "question": question,
            "context": context,
            "answers": example["answers"]["text"],
            "prompt_used": local_prompt,
            "generations": generations,
            "most_likely_answer": ml_answer,
            "accuracy": acc,
            "tbg_embedding": tbg_embedding,
            "slt_embedding": slt_embedding,
        })

        # Periodic save
        if (i + 1) % 100 == 0:
            with open(output_file, "wb") as f:
                pickle.dump(data_store, f)
            logging.info(f"Saved checkpoint at {i + 1} samples. "
                         f"Running accuracy: {np.mean(accuracies):.4f}")

    # Final save
    with open(output_file, "wb") as f:
        pickle.dump(data_store, f)

    task_accuracy = np.mean(accuracies) if accuracies else 0.0
    logging.info(f"Finished. Saved {len(data_store)} samples to {output_file}")
    logging.info(f"Task accuracy: {task_accuracy:.4f} ({task_accuracy*100:.1f}%)")
    logging.info(f"Few-shot indices used: {prompt_indices}")


if __name__ == "__main__":
    main()
