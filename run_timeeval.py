#!/usr/bin/env python3
"""Run the GLOutlier pipeline on one TimeEval canonical dataset.

The industrial-dataset scripts in the original release contain SWaT-specific
sensor names. This entry point performs the same generic stages for TimeEval:
discretization, local-factor partitioning, association-rule detection, AT-LSH,
and union of global/local detections. Labels are read only after prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def read_timeeval(path):
    frame = pd.read_csv(path)
    if frame.shape[1] < 3:
        raise ValueError(f"{path} is not a multivariate TimeEval CSV")
    label_col = "is_anomaly" if "is_anomaly" in frame.columns else frame.columns[-1]
    labels = pd.to_numeric(frame[label_col], errors="raise").fillna(0).astype(int).to_numpy()
    feature_cols = [c for c in frame.columns[1:] if c != label_col]
    values = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
    return values, labels, feature_cols


def discretize(fit_values, test_values):
    fit_out = np.zeros_like(fit_values, dtype=np.int64)
    test_out = np.zeros_like(test_values, dtype=np.int64)
    for j in range(fit_values.shape[1]):
        unique = np.unique(fit_values[:, j])
        if len(unique) <= 10:
            states = np.sort(unique)
            fit_out[:, j] = np.searchsorted(states, fit_values[:, j])
            indices = np.searchsorted(states, test_values[:, j]).clip(0, len(states) - 1)
            # Map unseen categorical/numeric values to the closest training state.
            left = np.maximum(indices - 1, 0)
            choose_left = np.abs(test_values[:, j] - states[left]) < np.abs(test_values[:, j] - states[indices])
            indices[choose_left] = left[choose_left]
            test_out[:, j] = indices
        else:
            bins = min(4, len(unique))
            encoder = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
            fit_out[:, j] = encoder.fit_transform(fit_values[:, [j]]).reshape(-1).astype(int)
            test_out[:, j] = encoder.transform(test_values[:, [j]]).reshape(-1).astype(int)
    return fit_out, test_out


def partition(fit_states, test_states, local_factor):
    local_fit = np.full_like(fit_states, -1)
    local_test = np.full_like(test_states, -1)
    global_states = {}
    for j in range(fit_states.shape[1]):
        values, counts = np.unique(fit_states[:, j], return_counts=True)
        support = dict(zip(values.tolist(), (counts / len(fit_states)).tolist()))
        common = {state for state, sup in support.items() if local_factor < sup < 0.99}
        rare = set(support) - common
        if common and rare:
            global_states[j] = common
            local_fit[:, j] = np.where(np.isin(fit_states[:, j], list(rare)), fit_states[:, j], -1)
            local_test[:, j] = np.where(np.isin(test_states[:, j], list(rare)), test_states[:, j], -1)
    keep = np.flatnonzero(np.any(local_fit != -1, axis=0))
    if len(keep) == 0:
        # A valid fallback for datasets without a > local_factor state: all
        # discretized channels are passed to AT-LSH and AR has no detections.
        return fit_states, test_states, {}
    return local_fit[:, keep], local_test[:, keep], global_states


def rule_tolerances(states, rule):
    left_col, left_value, right_col, right_value = rule
    tolerances = {"s": 0, "i": 0}
    context, run, active_context = "i", 0, "i"
    for row in states:
        left = row[left_col] == left_value
        right = row[right_col] == right_value
        if left and not right:
            if run == 0:
                active_context = context
            run += 1
            tolerances[active_context] = max(tolerances[active_context], run)
        else:
            run = 0
            context = "s" if left and right else "i"
    return tolerances["s"], tolerances["i"]


def association_rule_predictions(fit_states, test_states, global_states,
                                 min_support=0.7, min_confidence=0.9):
    rules = []
    n = len(fit_states)
    for left_col, left_values in global_states.items():
        for left_value in left_values:
            antecedent = fit_states[:, left_col] == left_value
            antecedent_count = int(antecedent.sum())
            if antecedent_count / n < min_support:
                continue
            for right_col in range(fit_states.shape[1]):
                if right_col == left_col:
                    continue
                for right_value in np.unique(fit_states[:, right_col]):
                    joint = antecedent & (fit_states[:, right_col] == right_value)
                    if joint.sum() / antecedent_count >= min_confidence:
                        rule = (left_col, left_value, right_col, right_value)
                        rules.append(rule + rule_tolerances(fit_states, rule))
    pred = np.zeros(len(test_states), dtype=np.int8)
    for left_col, left_value, right_col, right_value, tol_s, tol_i in rules:
        context, run, active_context = "i", 0, "i"
        for i, row in enumerate(test_states):
            left = row[left_col] == left_value
            right = row[right_col] == right_value
            if left and not right:
                if run == 0:
                    active_context = context
                run += 1
                tolerance = tol_s if active_context == "s" else tol_i
                if run > tolerance:
                    pred[i] = 1
            else:
                run = 0
                context = "s" if left and right else "i"
    return pred, len(rules)


def keep_minimum_runs(pred, minimum=10):
    result = np.zeros_like(pred)
    start = None
    for i, value in enumerate(np.r_[pred, 0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= minimum:
                result[start:i] = 1
            start = None
    return result


def write_atr_data(path, train, test, labels, window=100):
    path.mkdir(parents=True, exist_ok=True)
    remainder = len(test) % window
    padding = 0 if remainder == 0 else window - remainder
    if padding:
        test = np.vstack([test, np.repeat(test[-1:, :], padding, axis=0)])
        labels = np.r_[labels, np.zeros(padding, dtype=int)]
    columns = [f"x{i}" for i in range(train.shape[1])]
    pd.DataFrame(train, columns=columns).to_csv(path / "train.csv")
    pd.DataFrame(test, columns=columns).to_csv(path / "test.csv")
    pd.DataFrame({"is_anomaly": labels}).to_csv(path / "test_label.csv")
    return padding


def metrics(labels, pred):
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision_pct": precision * 100, "recall_pct": recall * 100,
            "f1_pct": f1 * 100, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--learning-type", choices=("semi-supervised", "unsupervised"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-factor", type=float, default=0.5)
    parser.add_argument("--support", type=float, default=0.7)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_values, labels, columns = read_timeeval(args.test)
    if args.train:
        fit_values, _, train_columns = read_timeeval(args.train)
        if train_columns != columns:
            raise ValueError("TimeEval train/test feature columns differ")
        fit_source = "TimeEval .train.csv"
    else:
        if args.learning_type != "unsupervised":
            raise ValueError("semi-supervised dataset is missing its .train.csv")
        fit_values = test_values
        fit_source = "unlabeled values from TimeEval .test.csv (unsupervised protocol)"

    fit_states, test_states = discretize(fit_values, test_values)
    local_fit, local_test, global_states = partition(
        fit_states, test_states, args.local_factor)
    global_pred, rule_count = association_rule_predictions(
        fit_states, test_states, global_states, args.support, args.confidence)

    atr_data = args.output_dir / "atr_data"
    padding = write_atr_data(atr_data, local_fit, local_test, labels)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{args.collection}-{args.dataset}")
    checkpoint_dir = args.output_dir / "checkpoints"
    atr_main = Path(__file__).resolve().parent / "ATR" / "main.py"
    common = [sys.executable, str(atr_main), "--dataset", safe_name,
              "--data_path", str(atr_data.resolve()), "--input_c", str(local_fit.shape[1]),
              "--output_c", str(local_fit.shape[1]), "--batch_size", str(args.batch_size),
              "--num_epochs", str(args.epochs), "--model_save_path", str(checkpoint_dir.resolve()),
              "--anormly_ratio", "1.0"]
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for AT-LSH") from exc
    subprocess.run(common + ["--mode", "train"], check=True, cwd=args.output_dir, env=env)
    subprocess.run(common + ["--mode", "test"], check=True, cwd=args.output_dir, env=env)
    result_path = args.output_dir / f"res_pickle_{safe_name}.txt"
    with result_path.open("rb") as f:
        local_pairs = pickle.load(f)
    local_pred = np.asarray([pair[1] for pair in local_pairs], dtype=np.int8)[:len(labels)]
    if padding and len(local_pred) < len(labels):
        raise RuntimeError("AT-LSH returned fewer predictions than the unpadded test series")
    unified = keep_minimum_runs(global_pred) | keep_minimum_runs(local_pred)
    scores = metrics(labels, unified)
    elapsed = time.time() - started
    output = pd.DataFrame({"is_anomaly": labels, "global_prediction": global_pred,
                           "local_prediction": local_pred, "prediction": unified})
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {"collection": args.collection, "dataset": args.dataset,
               "learning_type": args.learning_type, "fit_source": fit_source,
               "feature_count": len(columns), "local_feature_count": int(local_fit.shape[1]),
               "association_rule_count": rule_count, "runtime_s": elapsed, **scores}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
