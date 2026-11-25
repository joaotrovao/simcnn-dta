"""Train and evaluate SimCNN-DTA with k-fold cross-validation.

Runs can be interrupted and resumed: re-running the same command skips
finished folds and continues the current one from its last completed epoch.

Examples:
    python train.py --dataset davis
    python train.py --dataset davis --max-targets 20 --folds 3 --epochs 5
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from data import load_dataset
from metrics import evaluate
from model import build_model
from similarity import similarity_matrices


def make_batches(K1, K2, drug_idx, target_idx, y, batch_size, shuffle, seed):
    """Batches of (drug similarity vector, target similarity vector), affinity.

    The outer product is computed inside the model, so the full
    (n_pairs, n_drugs, n_targets) input tensor is never held in memory.
    """
    K1, K2 = tf.constant(K1), tf.constant(K2)
    ds = tf.data.Dataset.from_tensor_slices((drug_idx, target_idx, y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=seed)
    ds = ds.batch(batch_size).map(
        lambda d, t, y: ((tf.gather(K1, d), tf.gather(K2, t)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["davis", "kiba"], required=True)
    p.add_argument("--folds", type=int, default=5,
                   help="number of CV folds (paper: 3 and 5)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pool-strides", type=int, default=None,
                   help="max-pooling stride (default: 1 for Davis as in the paper, 2 for KIBA)")
    p.add_argument("--max-targets", type=int, default=None,
                   help="use only the first N targets, for quick runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    if args.pool_strides is None:
        args.pool_strides = 1 if args.dataset == "davis" else 2
    tf.keras.utils.set_random_seed(args.seed)

    data = load_dataset(args.dataset, args.max_targets)
    K1, K2 = similarity_matrices(data)
    print(f"{args.dataset}: {len(data.drugs)} drugs, {len(data.targets)} targets, "
          f"{len(data)} interactions")

    tag = args.dataset if args.max_targets is None else f"{args.dataset}_{args.max_targets}t"
    run_dir = (f"{args.checkpoint_dir}/{tag}_{args.folds}fold_{args.epochs}ep"
               f"_bs{args.batch_size}_pool{args.pool_strides}_seed{args.seed}")

    folds = KFold(args.folds, shuffle=True, random_state=args.seed)
    y_pred = np.zeros_like(data.y)
    fold_metrics = []

    for fold, (train, test) in enumerate(folds.split(data.y), start=1):
        print(f"\n--- Fold {fold}/{args.folds} "
              f"({len(train)} train / {len(test)} test) ---")
        fold_dir = f"{run_dir}/fold{fold}"
        pred_file = f"{fold_dir}/predictions.npy"

        if os.path.exists(pred_file):
            print("Already finished, loading saved predictions")
            y_pred[test] = np.load(pred_file)
        else:
            model = build_model(len(data.drugs), len(data.targets), args.pool_strides)
            if fold == 1:
                model.summary()

            batches = lambda idx, shuffle: make_batches(
                K1, K2, data.drug_idx[idx], data.target_idx[idx], data.y[idx],
                args.batch_size, shuffle, args.seed)
            # Saves model + optimizer state every epoch and restores it on restart
            backup = tf.keras.callbacks.BackupAndRestore(f"{fold_dir}/backup")
            model.fit(batches(train, True), epochs=args.epochs, verbose=2,
                      callbacks=[backup])

            y_pred[test] = model.predict(batches(test, False), verbose=0).ravel()
            np.save(pred_file, y_pred[test])

        scores = evaluate(data.y[test], y_pred[test], data.threshold)
        fold_metrics.append(scores)
        print("  ".join(f"{k}: {v:.4f}" for k, v in scores.items()))

    summary = {
        k: {"mean": float(np.mean([m[k] for m in fold_metrics])),
            "std": float(np.std([m[k] for m in fold_metrics]))}
        for k in fold_metrics[0]
    }
    print(f"\n=== {args.folds}-fold CV on {args.dataset} ===")
    for k, v in summary.items():
        print(f"{k:>5}: {v['mean']:.4f} (± {v['std']:.4f})")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{tag}_metrics.json", "w") as f:
        json.dump({"config": vars(args), "folds": fold_metrics, "summary": summary},
                  f, indent=2)

    # Every interaction is predicted exactly once, by the fold that held it out
    plt.figure(figsize=(5, 5))
    plt.scatter(y_pred, data.y, s=2, c="black", alpha=0.4)
    lims = [min(data.y.min(), y_pred.min()), max(data.y.max(), y_pred.max())]
    plt.plot(lims, lims, c="red")
    plt.xlabel("predicted")
    plt.ylabel("measured")
    plt.title(f"SimCNN-DTA on {args.dataset} ({args.folds}-fold CV)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/{tag}_scatter.png", dpi=150)


if __name__ == "__main__":
    main()
