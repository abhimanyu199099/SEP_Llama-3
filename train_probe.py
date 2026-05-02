"""Train semantic entropy probes with ID and OOD evaluation.

Modes:
  id     -- In-distribution evaluation on a single dataset
  ood    -- Train on one dataset, evaluate on another
  matrix -- Full N×N cross-dataset PR-AUC matrix

Strategies (--strategy, hidden-state probes only):
  concat     -- Concatenate top-k layer features, single logistic regression
  hard_vote  -- Per-layer probe on each top-k layer, majority vote
  soft_vote  -- Per-layer probe on each top-k layer, averaged probabilities
  meta       -- Per-layer probes on ALL layers; stack their output probs as
                features for a second logistic regression (meta-probe)

All logistic regressions use class_weight='balanced'.
Layer selection ranks layers by PR AUC (instead of AUROC).

Usage:
    python train_probe.py --mode id --dataset squad
    python train_probe.py --mode id --dataset squad --save_probe
    python train_probe.py --mode id --dataset squad --strategy hard_vote --top_k 8
    python train_probe.py --mode id --dataset squad --strategy meta
    python train_probe.py --mode ood --train_dataset squad --eval_dataset trivia_qa
    python train_probe.py --mode matrix
"""
import os
import pickle
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    average_precision_score, f1_score, recall_score, precision_score,
)
import logging

from common_utils import ALL_DATASETS, QA_DATASETS, OUTPUT_BASE, MODEL_NAME, NLI_MODEL

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Entropy Probes")
    parser.add_argument("--mode", choices=["id", "ood", "matrix"], default="id")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset for ID mode")
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="Training dataset for OOD mode")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Evaluation dataset for OOD mode")
    parser.add_argument("--feature_type", choices=["hidden", "lookback"], default="hidden",
                        help="'hidden' = TBG/SLT embeddings; 'lookback' = Lookback Ratio features")
    parser.add_argument("--strategy",
                        choices=["concat", "hard_vote", "soft_vote", "meta"],
                        default="concat",
                        help="Probe strategy for hidden-state features (default: concat)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top layers to select by PR AUC (default: 10)")
    parser.add_argument("--save_probe", action="store_true",
                        help="Save trained probe to output/{dataset}/sep_probe_*.pkl "
                             "(ID mode only, used by inference_with_gate.py)")
    return parser.parse_args()


# ============================================================
# Shared metrics helper
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    """Return a dict of classification metrics."""
    return {
        'auroc':     roc_auc_score(y_true, y_prob),
        'pr_auc':    average_precision_score(y_true, y_prob),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'accuracy':  accuracy_score(y_true, y_pred),
    }


def print_metrics(metrics, indent=2):
    pad = ' ' * indent
    print(f"{pad}AUROC:     {metrics['auroc']:.4f}")
    print(f"{pad}PR AUC:    {metrics['pr_auc']:.4f}")
    print(f"{pad}F1:        {metrics['f1']:.4f}")
    print(f"{pad}Recall:    {metrics['recall']:.4f}")
    print(f"{pad}Precision: {metrics['precision']:.4f}")
    print(f"{pad}Accuracy:  {metrics['accuracy']:.4f}")


# ============================================================
# Probe scoring (used by inference_with_gate.py via saved bundle)
# ============================================================

def score_from_bundle(bundle, emb_sq):
    """Score one embedding against a saved probe bundle.

    Args:
        bundle:  dict loaded from sep_probe_*.pkl
        emb_sq:  Tensor (num_layers, hidden_dim) — squeezed TBG or SLT embedding

    Returns:
        float probability of hallucination ∈ [0, 1]
    """
    strategy = bundle.get('strategy', 'concat')

    if strategy == 'concat':
        layer_indices = bundle['layer_indices']
        feat = np.concatenate(
            [emb_sq[l].numpy() for l in layer_indices], axis=0
        )[np.newaxis, :]
        return float(bundle['clf'].predict_proba(feat)[0, 1])

    elif strategy in ('hard_vote', 'soft_vote'):
        layer_probes = bundle['layer_probes']
        probs = np.array([
            clf.predict_proba(emb_sq[l].numpy()[np.newaxis, :])[0, 1]
            for l, clf in layer_probes
        ])
        if strategy == 'hard_vote':
            return float((probs >= 0.5).mean())
        return float(probs.mean())

    elif strategy == 'meta':
        layer_probes = bundle['layer_probes']
        probs = np.array([
            clf.predict_proba(emb_sq[l].numpy()[np.newaxis, :])[0, 1]
            for l, clf in layer_probes
        ])
        return float(bundle['meta_clf'].predict_proba(probs[np.newaxis, :])[0, 1])

    raise ValueError(f"Unknown probe strategy: {strategy}")


# ============================================================
# Core utilities
# ============================================================

def best_split(entropy):
    """1D k-means (k=2): find threshold minimising within-cluster MSE."""
    ents = entropy.numpy() if isinstance(entropy, torch.Tensor) else entropy
    splits = np.linspace(1e-10, ents.max(), 100)
    best_mse, best_threshold = np.inf, splits[0]
    for split in splits:
        low  = ents[ents <  split]
        high = ents[ents >= split]
        if len(low) == 0 or len(high) == 0:
            continue
        mse = ((low - low.mean()) ** 2).sum() + ((high - high.mean()) ** 2).sum()
        if mse < best_mse:
            best_mse, best_threshold = mse, split
    return best_threshold


def select_top_k_layers(scores, k):
    """Return indices of the top-k layers by score (highest first)."""
    k = min(k, len(scores))
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


def _new_lr():
    return LogisticRegression(class_weight='balanced', max_iter=1000)


# ============================================================
# Data loading
# ============================================================

def load_dataset_features(dataset_name):
    """Load all_layers.pt and binarize entropy labels."""
    data_file = os.path.join(OUTPUT_BASE, dataset_name, "all_layers.pt")
    logging.info(f"Loading {data_file}...")
    data = torch.load(data_file, weights_only=False)

    X_tbg    = data['X_tbg']    # (N, num_layers, hidden_dim)
    X_slt    = data['X_slt']    # (N, num_layers, hidden_dim)
    entropy  = data['entropy']  # (N,)

    threshold = best_split(entropy)
    y = (entropy >= threshold).long().numpy()

    num_samples, num_layers, hidden_dim = X_tbg.shape
    logging.info(f"  {dataset_name}: {num_samples} samples, {num_layers} layers, "
                 f"{hidden_dim}-dim, threshold={threshold:.4f}, "
                 f"low={np.sum(y==0)}, high={np.sum(y==1)}")
    return X_tbg, X_slt, y, threshold, entropy


def load_lookback_features(dataset_name):
    """Load lookback_features.pt and binarize entropy labels."""
    data_file = os.path.join(OUTPUT_BASE, dataset_name, "lookback_features.pt")
    logging.info(f"Loading {data_file} ...")
    data = torch.load(data_file, weights_only=False)

    X_lb    = data['X_lookback']  # (N, num_layers * num_heads)
    entropy = data['entropy']

    threshold = best_split(entropy)
    y = (entropy >= threshold).long().numpy()

    logging.info(f"  {dataset_name}: {X_lb.shape[0]} samples, "
                 f"{X_lb.shape[1]} lookback features, "
                 f"threshold={threshold:.4f}, low={np.sum(y==0)}, high={np.sum(y==1)}")
    return X_lb.numpy(), y, threshold, entropy


# ============================================================
# Per-layer sweep (PR AUC)
# ============================================================

def sweep_layers_on_split(X_np, y, num_layers, token_name=""):
    """Per-layer PR AUC sweep with internal train/val split."""
    pr_aucs = []
    N = len(y)
    idx = np.arange(N)
    idx_tv, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.125, random_state=42, stratify=y[idx_tv])

    for layer_idx in range(num_layers):
        clf = _new_lr()
        clf.fit(X_np[layer_idx][idx_tr], y[idx_tr])
        score = average_precision_score(
            y[idx_va], clf.predict_proba(X_np[layer_idx][idx_va])[:, 1])
        pr_aucs.append(score)
        if layer_idx % 5 == 0 and token_name:
            logging.info(f"  {token_name} Layer {layer_idx:2d}: PR AUC = {score:.4f}")

    return pr_aucs, (idx_tr, idx_va, idx_te)


def sweep_layers_full_train(X_train_np, y_train, X_eval_np, y_eval, num_layers):
    """Per-layer PR AUC: train on full train split, evaluate on full eval split."""
    pr_aucs = []
    for layer_idx in range(num_layers):
        clf = _new_lr()
        clf.fit(X_train_np[layer_idx], y_train)
        score = average_precision_score(
            y_eval, clf.predict_proba(X_eval_np[layer_idx])[:, 1])
        pr_aucs.append(score)
    return pr_aucs


# ============================================================
# Strategy: concat (top-k layers, single LR)
# ============================================================

def train_concat_id(X_np, y, top_k_indices, splits):
    idx_tr, idx_va, idx_te = splits
    X_tr = np.concatenate([X_np[l][idx_tr] for l in top_k_indices], axis=1)
    X_va = np.concatenate([X_np[l][idx_va] for l in top_k_indices], axis=1)
    X_te = np.concatenate([X_np[l][idx_te] for l in top_k_indices], axis=1)

    clf = _new_lr()
    clf.fit(X_tr, y[idx_tr])

    val_pr_auc = average_precision_score(y[idx_va], clf.predict_proba(X_va)[:, 1])
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return {
        'strategy':      'concat',
        'clf':           clf,
        'layer_indices': top_k_indices,
        'val_pr_auc':    val_pr_auc,
        'metrics':       compute_metrics(y[idx_te], y_pred, y_prob),
        'y_test':        y[idx_te],
        'y_pred':        y_pred,
    }


def train_concat_ood(X_train_np, y_train, X_eval_np, y_eval, top_k_indices):
    X_tr = np.concatenate([X_train_np[l] for l in top_k_indices], axis=1)
    X_ev = np.concatenate([X_eval_np[l]  for l in top_k_indices], axis=1)
    clf = _new_lr()
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_ev)
    y_prob = clf.predict_proba(X_ev)[:, 1]
    return compute_metrics(y_eval, y_pred, y_prob)


# ============================================================
# Strategy: hard_vote / soft_vote (per-layer probes, top-k)
# ============================================================

def train_voting_id(X_np, y, top_k_indices, splits, vote_mode):
    idx_tr, idx_va, idx_te = splits
    layer_probes = []
    for l in top_k_indices:
        clf = _new_lr()
        clf.fit(X_np[l][idx_tr], y[idx_tr])
        layer_probes.append((l, clf))

    def _aggregate(probs_2d):
        # probs_2d: (k, n_samples)
        if vote_mode == 'hard_vote':
            return (probs_2d >= 0.5).astype(float).mean(axis=0)
        return probs_2d.mean(axis=0)

    val_probs = np.array([clf.predict_proba(X_np[l][idx_va])[:, 1]
                          for l, clf in layer_probes])
    val_pr_auc = average_precision_score(y[idx_va], _aggregate(val_probs))

    test_probs = np.array([clf.predict_proba(X_np[l][idx_te])[:, 1]
                           for l, clf in layer_probes])
    y_prob = _aggregate(test_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        'strategy':      vote_mode,
        'layer_probes':  layer_probes,
        'layer_indices': top_k_indices,
        'val_pr_auc':    val_pr_auc,
        'metrics':       compute_metrics(y[idx_te], y_pred, y_prob),
        'y_test':        y[idx_te],
        'y_pred':        y_pred,
    }


def train_voting_ood(X_train_np, y_train, X_eval_np, y_eval, top_k_indices, vote_mode):
    layer_probes = []
    for l in top_k_indices:
        clf = _new_lr()
        clf.fit(X_train_np[l], y_train)
        layer_probes.append((l, clf))

    probs = np.array([clf.predict_proba(X_eval_np[l])[:, 1]
                      for l, clf in layer_probes])
    if vote_mode == 'hard_vote':
        y_prob = (probs >= 0.5).astype(float).mean(axis=0)
    else:
        y_prob = probs.mean(axis=0)
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(y_eval, y_pred, y_prob)


# ============================================================
# Strategy: meta (all-layer probes → meta LR)
# ============================================================

def train_meta_id(X_np, y, num_layers, splits):
    """
    Train one LR per layer on the train split.
    Collect their val-set probabilities → train meta LR on val probs.
    Evaluate on test set.
    No data leakage: layer probes never see val; meta LR never sees test.
    """
    idx_tr, idx_va, idx_te = splits

    layer_probes = []
    val_probs  = []
    test_probs = []

    for l in range(num_layers):
        clf = _new_lr()
        clf.fit(X_np[l][idx_tr], y[idx_tr])
        val_probs.append(clf.predict_proba(X_np[l][idx_va])[:, 1])
        test_probs.append(clf.predict_proba(X_np[l][idx_te])[:, 1])
        layer_probes.append((l, clf))

    X_meta_val  = np.array(val_probs).T   # (n_val,  num_layers)
    X_meta_test = np.array(test_probs).T  # (n_test, num_layers)

    meta_clf = _new_lr()
    meta_clf.fit(X_meta_val, y[idx_va])

    y_pred = meta_clf.predict(X_meta_test)
    y_prob = meta_clf.predict_proba(X_meta_test)[:, 1]

    val_pr_auc = average_precision_score(
        y[idx_va], meta_clf.predict_proba(X_meta_val)[:, 1])

    return {
        'strategy':     'meta',
        'layer_probes': layer_probes,
        'meta_clf':     meta_clf,
        'val_pr_auc':   val_pr_auc,
        'metrics':      compute_metrics(y[idx_te], y_pred, y_prob),
        'y_test':       y[idx_te],
        'y_pred':       y_pred,
    }


def train_meta_ood(X_train_np, y_train, X_eval_np, y_eval, num_layers):
    layer_probes = []
    train_probs  = []
    eval_probs   = []

    for l in range(num_layers):
        clf = _new_lr()
        clf.fit(X_train_np[l], y_train)
        train_probs.append(clf.predict_proba(X_train_np[l])[:, 1])
        eval_probs.append(clf.predict_proba(X_eval_np[l])[:, 1])
        layer_probes.append((l, clf))

    X_meta_train = np.array(train_probs).T
    X_meta_eval  = np.array(eval_probs).T

    meta_clf = _new_lr()
    meta_clf.fit(X_meta_train, y_train)

    y_pred = meta_clf.predict(X_meta_eval)
    y_prob = meta_clf.predict_proba(X_meta_eval)[:, 1]
    return compute_metrics(y_eval, y_pred, y_prob)


# ============================================================
# Mode: In-Distribution
# ============================================================

def main_id(dataset_name, strategy='concat', top_k=10, save_probe=False):
    X_tbg, X_slt, y, threshold, entropy = load_dataset_features(dataset_name)
    num_samples, num_layers, hidden_dim = X_tbg.shape

    results = {}
    for token_name, X_raw in [("TBG", X_tbg), ("SLT", X_slt)]:
        logging.info(f"\n{'='*40}\n[{dataset_name}] {token_name}  strategy={strategy}\n{'='*40}")

        X_np = X_raw.numpy().transpose(1, 0, 2)  # (num_layers, N, hidden_dim)

        pr_aucs, splits = sweep_layers_on_split(X_np, y, num_layers, token_name)
        top_k_indices   = select_top_k_layers(pr_aucs, top_k)
        logging.info(f"{token_name} top-{top_k} layers (PR AUC): {sorted(top_k_indices)}")

        if strategy == 'concat':
            result = train_concat_id(X_np, y, top_k_indices, splits)
        elif strategy in ('hard_vote', 'soft_vote'):
            result = train_voting_id(X_np, y, top_k_indices, splits, strategy)
        elif strategy == 'meta':
            result = train_meta_id(X_np, y, num_layers, splits)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result['per_layer_pr_aucs'] = pr_aucs
        result['top_k_indices']     = top_k_indices
        results[token_name] = result

    # ---- Print results ----
    print(f"\n{'='*60}")
    print(f"SEP RESULTS — {dataset_name} (In-Distribution)  strategy={strategy}")
    print(f"{'='*60}")
    print(f"Model:        {MODEL_NAME}")
    print(f"NLI Model:    {NLI_MODEL}")
    print(f"SE threshold: {threshold:.4f} (best_split)")
    print(f"Samples:      {num_samples}")
    print(f"Class dist:   low={np.sum(y==0)}, high={np.sum(y==1)}")
    print(f"Top-k layers: {top_k}")

    for tn in ["TBG", "SLT"]:
        r = results[tn]
        print(f"\n--- {tn} (strategy={r['strategy']}) ---")
        print(f"  Val PR AUC:    {r['val_pr_auc']:.4f}")
        print(f"  Test metrics:")
        print_metrics(r['metrics'], indent=4)
        if 'layer_indices' in r:
            print(f"  Selected layers: {sorted(r['layer_indices'])}")

    print(f"\nPer-layer PR AUC:")
    print(f"  {'Layer':>5}  {'TBG':>8}  {'SLT':>8}")
    for i in range(num_layers):
        tbg_m = "*" if i in results['TBG']['top_k_indices'] else " "
        slt_m = "*" if i in results['SLT']['top_k_indices'] else " "
        print(f"  {i:>5}  {results['TBG']['per_layer_pr_aucs'][i]:>8.4f}{tbg_m} "
              f"{results['SLT']['per_layer_pr_aucs'][i]:>8.4f}{slt_m}")
    print("  (* = in top-k)")

    for tn in ["TBG", "SLT"]:
        r = results[tn]
        print(f"\nClassification Report ({tn}):")
        print(classification_report(r['y_test'], r['y_pred'],
                                    target_names=["Low SE", "High SE"]))
    print("=" * 60)

    # ---- Optional: save probe ----
    if save_probe:
        X_tbg_np = X_tbg.numpy().transpose(1, 0, 2)
        X_slt_np = X_slt.numpy().transpose(1, 0, 2)

        for tn, X_np in [("TBG", X_tbg_np), ("SLT", X_slt_np)]:
            r = results[tn]

            if strategy == 'concat':
                layer_indices = r['layer_indices']
                X_full = np.concatenate([X_np[l] for l in layer_indices], axis=1)
                clf_full = _new_lr()
                clf_full.fit(X_full, y)
                probe_bundle = {
                    'strategy':      'concat',
                    'clf':           clf_full,
                    'layer_indices': layer_indices,
                    'threshold':     threshold,
                    'token_type':    tn,
                    'dataset':       dataset_name,
                    'hidden_dim':    hidden_dim,
                    'num_layers':    num_layers,
                }

            elif strategy in ('hard_vote', 'soft_vote'):
                layer_probes_full = []
                for l in r['layer_indices']:
                    clf_l = _new_lr()
                    clf_l.fit(X_np[l], y)
                    layer_probes_full.append((l, clf_l))
                probe_bundle = {
                    'strategy':      strategy,
                    'layer_probes':  layer_probes_full,
                    'layer_indices': r['layer_indices'],
                    'threshold':     threshold,
                    'token_type':    tn,
                    'dataset':       dataset_name,
                    'hidden_dim':    hidden_dim,
                    'num_layers':    num_layers,
                }

            elif strategy == 'meta':
                # Retrain layer probes on full data; keep the val-trained meta clf.
                layer_probes_full = []
                for l in range(num_layers):
                    clf_l = _new_lr()
                    clf_l.fit(X_np[l], y)
                    layer_probes_full.append((l, clf_l))
                probe_bundle = {
                    'strategy':     'meta',
                    'layer_probes': layer_probes_full,
                    'meta_clf':     r['meta_clf'],
                    'threshold':    threshold,
                    'token_type':   tn,
                    'dataset':      dataset_name,
                    'hidden_dim':   hidden_dim,
                    'num_layers':   num_layers,
                }

            probe_path = os.path.join(OUTPUT_BASE, dataset_name, f"sep_probe_{tn}.pkl")
            with open(probe_path, "wb") as f:
                pickle.dump(probe_bundle, f)
            logging.info(f"Saved {tn} probe → {probe_path}  "
                         f"(strategy={strategy}, threshold={threshold:.4f})")

    return results


# ============================================================
# Mode: Out-of-Distribution
# ============================================================

def main_ood(train_dataset, eval_dataset, strategy='concat', top_k=10):
    logging.info(f"OOD: train={train_dataset}, eval={eval_dataset}, strategy={strategy}")

    X_tbg_train, X_slt_train, y_train, thresh_train, _ = load_dataset_features(train_dataset)
    X_tbg_eval,  X_slt_eval,  y_eval,  thresh_eval,  _ = load_dataset_features(eval_dataset)
    num_layers = X_tbg_train.shape[1]

    results = {}
    for token_name, X_tr_raw, X_ev_raw in [
        ("TBG", X_tbg_train, X_tbg_eval),
        ("SLT", X_slt_train, X_slt_eval),
    ]:
        X_train_np = X_tr_raw.numpy().transpose(1, 0, 2)
        X_eval_np  = X_ev_raw.numpy().transpose(1, 0, 2)

        pr_aucs = sweep_layers_full_train(X_train_np, y_train, X_eval_np, y_eval, num_layers)
        top_k_indices = select_top_k_layers(pr_aucs, top_k)

        if strategy == 'concat':
            metrics = train_concat_ood(X_train_np, y_train, X_eval_np, y_eval, top_k_indices)
        elif strategy in ('hard_vote', 'soft_vote'):
            metrics = train_voting_ood(X_train_np, y_train, X_eval_np, y_eval,
                                       top_k_indices, strategy)
        elif strategy == 'meta':
            metrics = train_meta_ood(X_train_np, y_train, X_eval_np, y_eval, num_layers)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        results[token_name] = {
            'metrics':            metrics,
            'top_k_indices':      top_k_indices,
            'per_layer_pr_aucs':  pr_aucs,
        }

    print(f"\n{'='*60}")
    print(f"SEP OOD RESULTS — Train: {train_dataset} -> Eval: {eval_dataset}  strategy={strategy}")
    print(f"{'='*60}")
    print(f"Train threshold: {thresh_train:.4f}, Eval threshold: {thresh_eval:.4f}")
    print(f"Train: {len(y_train)} samples  |  Eval: {len(y_eval)} samples")
    for tn in ["TBG", "SLT"]:
        r = results[tn]
        print(f"\n--- {tn} ---")
        print_metrics(r['metrics'])
        print(f"  Top-{top_k} layers: {sorted(r['top_k_indices'])}")
    print("=" * 60)
    return results


# ============================================================
# Mode: Matrix (all cross-dataset pairs, concat strategy)
# ============================================================

def main_matrix(top_k=10):
    available = [ds for ds in ALL_DATASETS
                 if os.path.exists(os.path.join(OUTPUT_BASE, ds, "all_layers.pt"))]

    if len(available) < 2:
        logging.error(f"Need ≥2 datasets for matrix. Found: {available}")
        return

    logging.info(f"Matrix datasets: {available}")

    all_data = {}
    for ds in available:
        X_tbg, X_slt, y, threshold, entropy = load_dataset_features(ds)
        all_data[ds] = {
            'X_tbg': X_tbg, 'X_slt': X_slt, 'y': y,
            'threshold': threshold, 'entropy': entropy,
        }

    for token_name in ["TBG", "SLT"]:
        metric_matrix = {
            'pr_auc': np.zeros((len(available), len(available))),
            'f1':     np.zeros((len(available), len(available))),
            'recall': np.zeros((len(available), len(available))),
        }

        for i, train_ds in enumerate(available):
            X_tr_raw = all_data[train_ds][f'X_{token_name.lower()}']
            y_train  = all_data[train_ds]['y']
            num_layers = X_tr_raw.shape[1]
            X_train_np = X_tr_raw.numpy().transpose(1, 0, 2)

            for j, eval_ds in enumerate(available):
                X_ev_raw  = all_data[eval_ds][f'X_{token_name.lower()}']
                y_eval    = all_data[eval_ds]['y']
                X_eval_np = X_ev_raw.numpy().transpose(1, 0, 2)

                if train_ds == eval_ds:
                    pr_aucs, splits = sweep_layers_on_split(X_train_np, y_train, num_layers)
                    top_k_indices   = select_top_k_layers(pr_aucs, top_k)
                    r = train_concat_id(X_train_np, y_train, top_k_indices, splits)
                    m = r['metrics']
                else:
                    pr_aucs       = sweep_layers_full_train(
                        X_train_np, y_train, X_eval_np, y_eval, num_layers)
                    top_k_indices = select_top_k_layers(pr_aucs, top_k)
                    m = train_concat_ood(X_train_np, y_train, X_eval_np, y_eval, top_k_indices)

                metric_matrix['pr_auc'][i][j] = m['pr_auc']
                metric_matrix['f1'][i][j]     = m['f1']
                metric_matrix['recall'][i][j] = m['recall']
                logging.info(f"  {token_name} {train_ds:>10} -> {eval_ds:<10}: "
                             f"PR_AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}  "
                             f"Recall={m['recall']:.4f}")

        for metric_name, mat in metric_matrix.items():
            print(f"\n{'='*70}")
            print(f"CROSS-DATASET {metric_name.upper()} MATRIX — {token_name}  (top_k={top_k})")
            print(f"{'='*70}")
            print(f"Model: {MODEL_NAME} | NLI: {NLI_MODEL}")
            header = 'Train \\ Eval'.rjust(14)
            for ds in available:
                header += f"  {ds:>12}"
            print(header)
            print("-" * len(header))
            for i, train_ds in enumerate(available):
                row = f"{train_ds:>14}"
                for j in range(len(available)):
                    marker = " *" if i == j else "  "
                    row += f"  {mat[i][j]:>10.4f}{marker}"
                print(row)
            diag     = np.diag(mat)
            off_diag = mat[~np.eye(len(available), dtype=bool)]
            print(f"\n  ID mean:  {np.mean(diag):.4f}")
            print(f"  OOD mean: {np.mean(off_diag):.4f}")

        print(f"\n  SE thresholds:")
        for ds in available:
            t    = all_data[ds]['threshold']
            n    = len(all_data[ds]['y'])
            low  = np.sum(all_data[ds]['y'] == 0)
            high = np.sum(all_data[ds]['y'] == 1)
            print(f"    {ds:>12}: threshold={t:.4f}, N={n}, low={low}, high={high}")
    print(f"\n{'='*70}")


# ============================================================
# Lookback Lens probe
# ============================================================

def main_lookback_id(dataset_name):
    X_lb, y, threshold, entropy = load_lookback_features(dataset_name)

    idx = np.arange(len(y))
    idx_tv, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.125, random_state=42, stratify=y[idx_tv])

    clf = _new_lr()
    clf.fit(X_lb[idx_tr], y[idx_tr])

    val_pr_auc = average_precision_score(y[idx_va], clf.predict_proba(X_lb[idx_va])[:, 1])
    y_pred = clf.predict(X_lb[idx_te])
    y_prob = clf.predict_proba(X_lb[idx_te])[:, 1]
    metrics = compute_metrics(y[idx_te], y_pred, y_prob)

    print(f"\n{'='*60}")
    print(f"LOOKBACK LENS RESULTS — {dataset_name} (In-Distribution)")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}  |  NLI: {NLI_MODEL}")
    print(f"SE threshold: {threshold:.4f}  |  Samples: {len(y)}")
    print(f"Class dist: low={np.sum(y==0)}, high={np.sum(y==1)}")
    print(f"Feature dim: {X_lb.shape[1]}  (num_layers × num_heads)")
    print(f"\nVal PR AUC: {val_pr_auc:.4f}")
    print(f"Test metrics:")
    print_metrics(metrics)
    print(f"\nClassification Report:")
    print(classification_report(y[idx_te], y_pred, target_names=["Low SE", "High SE"]))
    print("=" * 60)


def main_lookback_ood(train_dataset, eval_dataset):
    X_tr, y_tr, thresh_tr, _ = load_lookback_features(train_dataset)
    X_ev, y_ev, thresh_ev, _ = load_lookback_features(eval_dataset)

    clf = _new_lr()
    clf.fit(X_tr, y_tr)
    y_pred   = clf.predict(X_ev)
    y_prob   = clf.predict_proba(X_ev)[:, 1]
    metrics  = compute_metrics(y_ev, y_pred, y_prob)

    print(f"\n{'='*60}")
    print(f"LOOKBACK LENS OOD — Train: {train_dataset} -> Eval: {eval_dataset}")
    print(f"{'='*60}")
    print(f"Train threshold: {thresh_tr:.4f}  |  Eval threshold: {thresh_ev:.4f}")
    print_metrics(metrics)
    print("=" * 60)


def main_lookback_matrix():
    available = [ds for ds in ALL_DATASETS
                 if os.path.exists(os.path.join(OUTPUT_BASE, ds, "lookback_features.pt"))]

    if len(available) < 2:
        logging.error(f"Need ≥2 datasets. Found: {available}")
        return

    all_data = {ds: {} for ds in available}
    for ds in available:
        X, y, threshold, _ = load_lookback_features(ds)
        all_data[ds] = {'X': X, 'y': y, 'threshold': threshold}

    pr_auc_matrix = np.zeros((len(available), len(available)))
    f1_matrix     = np.zeros((len(available), len(available)))

    for i, train_ds in enumerate(available):
        for j, eval_ds in enumerate(available):
            X_tr, y_tr = all_data[train_ds]['X'], all_data[train_ds]['y']
            X_ev, y_ev = all_data[eval_ds]['X'],  all_data[eval_ds]['y']
            if train_ds == eval_ds:
                idx = np.arange(len(y_tr))
                idx_tv, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_tr)
                idx_tr2, _     = train_test_split(idx_tv, test_size=0.125, random_state=42, stratify=y_tr[idx_tv])
                clf = _new_lr()
                clf.fit(X_tr[idx_tr2], y_tr[idx_tr2])
                y_pred = clf.predict(X_tr[idx_te])
                y_prob = clf.predict_proba(X_tr[idx_te])[:, 1]
                m = compute_metrics(y_tr[idx_te], y_pred, y_prob)
            else:
                clf = _new_lr()
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_ev)
                y_prob = clf.predict_proba(X_ev)[:, 1]
                m = compute_metrics(y_ev, y_pred, y_prob)
            pr_auc_matrix[i][j] = m['pr_auc']
            f1_matrix[i][j]     = m['f1']
            logging.info(f"  {train_ds:>12} -> {eval_ds:<12}: "
                         f"PR_AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")

    for name, mat in [("PR AUC", pr_auc_matrix), ("F1", f1_matrix)]:
        print(f"\n{'='*70}")
        print(f"LOOKBACK LENS — CROSS-DATASET {name} MATRIX")
        print(f"{'='*70}")
        header = 'Train \\ Eval'.rjust(14)
        for ds in available:
            header += f"  {ds:>12}"
        print(header)
        print("-" * len(header))
        for i, train_ds in enumerate(available):
            row = f"{train_ds:>14}"
            for j in range(len(available)):
                marker = " *" if i == j else "  "
                row += f"  {mat[i][j]:>10.4f}{marker}"
            print(row)
        diag     = np.diag(mat)
        off_diag = mat[~np.eye(len(available), dtype=bool)]
        print(f"\n  ID mean:  {np.mean(diag):.4f}")
        print(f"  OOD mean: {np.mean(off_diag):.4f}")
    print(f"\n{'='*70}")


# ============================================================
# Main dispatch
# ============================================================

def main():
    args = parse_args()

    if args.feature_type == "lookback":
        if args.mode == "id":
            if args.dataset is None:
                logging.error("--dataset required for ID mode"); return
            main_lookback_id(args.dataset)
        elif args.mode == "ood":
            if not (args.train_dataset and args.eval_dataset):
                logging.error("--train_dataset and --eval_dataset required"); return
            main_lookback_ood(args.train_dataset, args.eval_dataset)
        elif args.mode == "matrix":
            main_lookback_matrix()
        return

    # Hidden-state (TBG / SLT) probe
    if args.mode == "id":
        if args.dataset is None:
            logging.error("--dataset required for ID mode"); return
        main_id(args.dataset, strategy=args.strategy, top_k=args.top_k,
                save_probe=args.save_probe)

    elif args.mode == "ood":
        if not (args.train_dataset and args.eval_dataset):
            logging.error("--train_dataset and --eval_dataset required"); return
        main_ood(args.train_dataset, args.eval_dataset,
                 strategy=args.strategy, top_k=args.top_k)

    elif args.mode == "matrix":
        main_matrix(top_k=args.top_k)


if __name__ == "__main__":
    main()
