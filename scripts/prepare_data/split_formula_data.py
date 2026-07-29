"""Create deterministic splits for formula datasets."""

import argparse
import json
import os
import random


def split_records(records,
                  seed,
                  train_ratio=0.70,
                  valid_ratio=0.15,
                  test_ratio=0.15):
    if not records:
        raise ValueError("input dataset is empty")
    if (train_ratio <= 0 or valid_ratio <= 0 or test_ratio < 0
            or train_ratio + valid_ratio + test_ratio > 1):
        raise ValueError("invalid train/valid/test ratios")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    train_end = round(len(indices) * train_ratio)
    valid_end = round(len(indices) * (train_ratio + valid_ratio))
    splits = {
        "train": [records[i] for i in indices[:train_end]],
        "valid": [records[i] for i in indices[train_end:valid_end]],
    }
    if test_ratio > 0:
        splits["test"] = [records[i] for i in indices[valid_end:]]
    elif valid_end != len(indices):
        raise ValueError("ratios must cover all records when test_ratio is 0")
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="set to 0 for pretraining data with no held-out test set")
    args = parser.parse_args()

    with open(args.input, "r") as file:
        records = json.load(file)
    splits = split_records(records, args.seed, args.train_ratio,
                           args.valid_ratio, args.test_ratio)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "source": os.path.abspath(args.input),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "counts": {
            name: len(values)
            for name, values in splits.items()
        },
    }
    for name, values in splits.items():
        with open(os.path.join(args.output_dir, f"{args.name}_{name}.json"),
                  "w") as file:
            json.dump(values, file, indent=2)
    with open(os.path.join(args.output_dir, f"{args.name}_manifest.json"),
              "w") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
