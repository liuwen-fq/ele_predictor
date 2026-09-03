#!/usr/bin/env python3
"""Evaluate ordinary and explainable anion-ratio predictions."""

import argparse
import json
import math


def metrics(records, key):
    pairs = [(float(row["anion_ratio"]), float(row[key])) for row in records
             if row.get("anion_ratio_mask", False) and row.get(key) is not None]
    if not pairs:
        raise ValueError(f"no labelled predictions for {key}")
    targets, predictions = zip(*pairs)
    errors = [prediction - target for target, prediction in pairs]
    mean_target = sum(targets) / len(targets)
    ss_res = sum(error * error for error in errors)
    ss_tot = sum((target - mean_target) ** 2 for target in targets)
    return {
        "prediction_key": key,
        "n": len(pairs),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(ss_res / len(errors)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mae-degradation", type=float, default=.01)
    args = parser.parse_args()
    records = json.loads(open(args.input).read())
    ordinary = metrics(records, "anion_ratio_pred")
    explained = metrics(records, "anion_ratio_explain_pred")
    degradation = explained["mae"] - ordinary["mae"]
    result = {
        "ordinary": ordinary,
        "explain": explained,
        "mae_degradation": degradation,
        "max_mae_degradation": args.max_mae_degradation,
        "analysis_eligible": degradation <= args.max_mae_degradation,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
