#!/usr/bin/env python3
"""Prepare a fixed synthetic set for a 4-D mechanism-conditioned diffusion run.

This utility deliberately does *not* construct or oracle-score formulas.  It
copies an existing Q synthetic set, z-scores its formula-level explanation
interaction proxy ``[I_sigma, I_r]``, and optionally breaks only the factual
formula--proxy pairing inside coarse (sigma, r, solvent-count) cells.

The true and shuffled outputs therefore have the same formulas, BOW targets,
sigma/r labels, mechanism marginal distribution and 30,000-row budget.  Their
only designed difference is whether a formula retains its own interaction
proxy.  The shared empirical mechanism prior is always built from the original
matched source set, so it cannot itself reveal the shuffle arm at inference.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


SIGMA_EDGES = np.asarray([0., 5., 10., 15., 20., 25., 30., 40., 100.])
RATIO_EDGES = np.asarray([0., .2, .4, .6, .8, 1.])


def mechanism_cell(item, record):
    """Return the same coarse cell used by the previous Q/I data selector."""
    sigma = float(torch.exp(item["conductivity"] * 8.3344 - 4.6002))
    ratio = float(item["anion_ratio"])
    sigma_bin = min(np.digitize(sigma, SIGMA_EDGES) - 1, len(SIGMA_EDGES) - 2)
    ratio_bin = min(np.digitize(ratio, RATIO_EDGES) - 1, len(RATIO_EDGES) - 2)
    n_solvents = len(record["solvents"])
    solvent_cohort = "1" if n_solvents == 1 else ("2_3" if n_solvents <= 3 else "4_6")
    return int(sigma_bin), int(ratio_bin), solvent_cohort


def shuffled_partner_indices(data, records, seed):
    """Block-permute complete [I_sigma, I_r] pairs with no fixed pair for n>1."""
    cells = defaultdict(list)
    for index, (item, record) in enumerate(zip(data, records)):
        cells[mechanism_cell(item, record)].append(index)

    rng = random.Random(seed)
    partner = list(range(len(data)))
    shuffled_cells, moved = 0, 0
    for cell, indices in cells.items():
        if len(indices) < 2:
            continue
        permutation = list(indices)
        # A random derangement means no formula in a multi-row cell keeps its
        # own pair.  This makes M-shuffle a stronger falsification control than
        # a vanilla permutation that can leave many pairings unchanged.
        for _ in range(1000):
            rng.shuffle(permutation)
            if all(left != right for left, right in zip(indices, permutation)):
                break
        else:
            raise RuntimeError(f"could not derange mechanism labels in cell {cell}")
        for target, source in zip(indices, permutation):
            partner[target] = source
        shuffled_cells += 1
        moved += len(indices)
    return partner, {
        "num_cells": len(cells),
        "num_cells_with_block_shuffle": shuffled_cells,
        "num_rows_with_shuffled_pair": moved,
        "num_singleton_rows_kept": len(data) - moved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", required=True,
                        help="Existing Q generator_data directory with mechanism_emb.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("true", "shuffle_within_cell"),
                        required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    source = Path(args.source_data_dir).resolve()
    output = Path(args.output_dir).resolve()
    source_pkl = source / "processed_data.pkl"
    source_json = source / "synthetic_data.json"
    source_dataset = source / "dataset_config.yaml"
    source_metadata = source / "synthetic_metadata.json"
    for path in (source_pkl, source_json, source_dataset, source_metadata):
        if not path.is_file():
            parser.error(f"missing required source file: {path}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # These are generated data objects, not external/untrusted checkpoints.
    data = torch.load(source_pkl, weights_only=False)
    records = json.loads(source_json.read_text())
    if len(data) != len(records) or not data:
        raise RuntimeError("source processed_data.pkl and synthetic_data.json are not aligned")
    if any("mechanism_emb" not in item for item in data):
        raise RuntimeError("source synthetic data lacks mechanism_emb; rebuild Q with explanations")

    raw_mechanism = torch.stack([
        torch.as_tensor(item["mechanism_emb"], dtype=torch.float32).reshape(-1)
        for item in data
    ])
    if raw_mechanism.shape != (len(data), 2) or not torch.isfinite(raw_mechanism).all():
        raise RuntimeError(
            "expected finite mechanism_emb shape [num_formulas, 2], got "
            f"{tuple(raw_mechanism.shape)}")
    mechanism_mean = raw_mechanism.mean(dim=0)
    mechanism_std = raw_mechanism.std(dim=0, unbiased=False)
    if torch.any(mechanism_std <= 1e-8):
        raise RuntimeError("at least one mechanism feature has near-zero variance")
    standardized_source = (raw_mechanism - mechanism_mean) / mechanism_std

    conditions = torch.stack([
        torch.as_tensor(item["conductivity"], dtype=torch.float32).reshape(-1)[0]
        for item in data
    ])
    ratios = torch.stack([
        torch.as_tensor(item["anion_ratio"], dtype=torch.float32).reshape(-1)[0]
        for item in data
    ])
    conditions = torch.stack((conditions, ratios), dim=1)
    condition_mean = conditions.mean(dim=0)
    condition_std = conditions.std(dim=0, unbiased=False).clamp_min(1e-8)

    if args.mode == "true":
        partner = list(range(len(data)))
        shuffle_info = {
            "num_cells": None,
            "num_cells_with_block_shuffle": 0,
            "num_rows_with_shuffled_pair": 0,
            "num_singleton_rows_kept": len(data),
        }
    else:
        partner, shuffle_info = shuffled_partner_indices(data, records, args.seed)

    selected_mechanism = standardized_source[torch.tensor(partner, dtype=torch.long)]
    prepared_data = []
    prepared_records = []
    for index, item in enumerate(data):
        prepared = dict(item)
        prepared["mechanism_emb"] = selected_mechanism[index].clone()
        prepared_data.append(prepared)
        # Retain the formula's original explanation proxy (if present) for
        # attribution audit, while recording the z-scored pair actually seen
        # by the diffusion loss.  In the shuffle arm these intentionally
        # differ, so synthetic_data.json cannot be mistaken for the training
        # mechanism assignment.
        record = dict(records[index])
        record["mechanism_conditioning"] = {
            "training_mechanism_emb_zscore": [
                float(value) for value in selected_mechanism[index]],
            "pairing_mode": args.mode,
        }
        prepared_records.append(record)
    torch.save(prepared_data, output / "processed_data.pkl")
    (output / "synthetic_data.json").write_text(
        json.dumps(prepared_records, indent=2) + "\n")

    # The prior is intentionally generated from *true source* labels in both
    # arms.  At sampling it exposes only p(I_sigma, I_r | sigma, r), never the
    # unknown interaction of a just-generated formula.
    torch.save({
        "schema_version": 1,
        "conditions": conditions,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "mechanism_emb_zscore": standardized_source,
        "mechanism_raw_mean": mechanism_mean,
        "mechanism_raw_std": mechanism_std,
        "feature_names": ["I_sigma", "I_r"],
        "source_data_dir": str(source),
        "prior_source": "true_formula_mechanism_pairs_before_optional_shuffle",
    }, output / "mechanism_prior.pt")

    dataset_config = yaml.safe_load(source_dataset.read_text())
    dataset_config["save_dir"] = str(output)
    (output / "dataset_config.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False))

    source_meta = json.loads(source_metadata.read_text())
    source_meta["synthetic_data_json"] = str(output / "synthetic_data.json")
    source_meta["mechanism_conditioning"] = {
        "enabled": True,
        "mode": args.mode,
        "source_data_dir": str(source),
        "training_feature": "mechanism_emb",
        "feature_names": ["I_sigma", "I_r"],
        "feature_transform": "global_train_set_zscore",
        "raw_mean": [float(value) for value in mechanism_mean],
        "raw_std": [float(value) for value in mechanism_std],
        "inference_prior": "conditional_empirical_knn_from_true_source_pairs",
        "shuffle": shuffle_info,
        "shuffle_seed": args.seed if args.mode == "shuffle_within_cell" else None,
        "pairing_preserved": args.mode == "true",
    }
    (output / "synthetic_metadata.json").write_text(
        json.dumps(source_meta, indent=2) + "\n")
    (output / "mechanism_conditioning.json").write_text(json.dumps({
        "mode": args.mode,
        "source_data_dir": str(source),
        "output_dir": str(output),
        "num_formulas": len(data),
        "same_formula_order_as_source": True,
        "mechanism_feature_names": ["I_sigma", "I_r"],
        "mechanism_raw_mean": [float(value) for value in mechanism_mean],
        "mechanism_raw_std": [float(value) for value in mechanism_std],
        "condition_mean": [float(value) for value in condition_mean],
        "condition_std": [float(value) for value in condition_std],
        "shared_inference_prior": str(output / "mechanism_prior.pt"),
        "shuffle": shuffle_info,
    }, indent=2) + "\n")
    print(
        f"Prepared {args.mode} mechanism-conditioned data: {len(data)} formulas -> {output}",
        flush=True)


if __name__ == "__main__":
    main()
