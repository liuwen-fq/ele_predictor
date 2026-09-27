#!/usr/bin/env python3
"""Prepare matched data for interaction-residual auxiliary supervision.

This script never builds or scores new formulas.  It starts from one completed
Q synthetic set, keeps its formulas, BOW targets, formula embeddings, and
two-dimensional (conductivity, anion-ratio) labels unchanged, then adds a
training-only ``mechanism_aux_emb`` target.

The target is the explanation-head pair interaction after centering within a
coarse (sigma, r, solvent-count) cell.  Thus it represents the part of the
interaction proxy not explained by the primary conditional variables at the
resolution used by the counterfactual control.  ``shuffle_within_cell``
breaks only the formula--residual pairing; it preserves all marginals and
never changes a generation-time condition.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


SIGMA_EDGES = np.asarray([0., 5., 10., 15., 20., 25., 30., 40., 100.])
RATIO_EDGES = np.asarray([0., .2, .4, .6, .8, 1.])


def mechanism_cell(item, record):
    """Return the matched conditioning cell used for residualization/shuffle."""
    sigma = float(torch.exp(item["conductivity"] * 8.3344 - 4.6002))
    ratio = float(item["anion_ratio"])
    sigma_bin = min(np.digitize(sigma, SIGMA_EDGES) - 1,
                    len(SIGMA_EDGES) - 2)
    ratio_bin = min(np.digitize(ratio, RATIO_EDGES) - 1,
                    len(RATIO_EDGES) - 2)
    n_solvents = len(record["solvents"])
    solvent_cohort = "1" if n_solvents == 1 else (
        "2_3" if n_solvents <= 3 else "4_6")
    return int(sigma_bin), int(ratio_bin), solvent_cohort


def cell_index(data, records):
    cells = defaultdict(list)
    for index, (item, record) in enumerate(zip(data, records)):
        cells[mechanism_cell(item, record)].append(index)
    return cells


def conditional_residuals(raw_interaction, cells):
    """Subtract each cell mean while retaining the formula order."""
    residual = torch.empty_like(raw_interaction)
    cell_sizes = {}
    for cell, indices in cells.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        values = raw_interaction[index_tensor]
        residual[index_tensor] = values - values.mean(dim=0, keepdim=True)
        cell_sizes["|".join(map(str, cell))] = len(indices)
    return residual, cell_sizes


def deranged_partner_indices(cells, size, seed):
    """Block-permute complete two-value labels without fixed pairs for n>1."""
    rng = random.Random(seed)
    partner = list(range(size))
    shuffled_cells, moved = 0, 0
    for cell, indices in cells.items():
        if len(indices) < 2:
            continue
        permutation = list(indices)
        for _ in range(1000):
            rng.shuffle(permutation)
            if all(left != right for left, right in zip(indices, permutation)):
                break
        else:
            raise RuntimeError(
                f"could not derange mechanism auxiliary labels in cell {cell}")
        for target, source in zip(indices, permutation):
            partner[target] = source
        shuffled_cells += 1
        moved += len(indices)
    return partner, {
        "num_cells": len(cells),
        "num_cells_with_block_shuffle": shuffled_cells,
        "num_rows_with_shuffled_pair": moved,
        "num_singleton_rows_kept": size - moved,
    }


TARGET_COLUMNS = {
    "sigma": (0,),
    "r": (1,),
    "sigma_r": (0, 1),
}
TARGET_FEATURE_NAMES = {
    "sigma": ["I_sigma_conditional_residual"],
    "r": ["I_r_conditional_residual"],
    "sigma_r": ["I_sigma_conditional_residual",
                "I_r_conditional_residual"],
}


def atomic_torch_save(value, path):
    """Publish MA data only after its torch payload is complete."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path, text):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add factual or within-cell-shuffled interaction residual "
        "labels to a fixed synthetic generator dataset.")
    parser.add_argument("--source-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("true", "shuffle_within_cell"),
                        required=True)
    parser.add_argument(
        "--auxiliary-targets", choices=tuple(TARGET_COLUMNS),
        default="sigma_r",
        help="Which residual interaction labels supervise diffusion: sigma "
             "for I_sigma only, r for I_r only, or sigma_r for the factual "
             "two-label auxiliary task.")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source_data_dir).resolve()
    output = Path(args.output_dir).resolve()
    required = {
        "processed": source / "processed_data.pkl",
        "synthetic": source / "synthetic_data.json",
        "dataset": source / "dataset_config.yaml",
        "metadata": source / "synthetic_metadata.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} source file: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output}; use a new MA run "
            "directory rather than overwriting training data")
    output.mkdir(parents=True, exist_ok=True)

    try:
        data = torch.load(required["processed"], weights_only=False)
    except EOFError as error:
        raise RuntimeError(
            "source processed_data.pkl is empty or incompletely written: "
            f"{required['processed']}. Rebuild the shared source data before "
            "preparing MA labels.") from error
    records = json.loads(required["synthetic"].read_text())
    if not data or len(data) != len(records):
        raise RuntimeError(
            "processed_data.pkl and synthetic_data.json must contain the "
            "same nonzero number of formulas")
    if any("mechanism_emb" not in item for item in data):
        raise RuntimeError(
            "source synthetic data has no mechanism_emb. Rebuild its matched "
            "Q data with explanation interaction metadata before running MA.")

    raw_interaction = torch.stack([
        torch.as_tensor(item["mechanism_emb"], dtype=torch.float32).reshape(-1)
        for item in data
    ])
    if raw_interaction.shape != (len(data), 2):
        raise RuntimeError(
            "mechanism_emb must contain exactly [I_sigma, I_r] per formula; "
            f"got {tuple(raw_interaction.shape)}")
    if not torch.isfinite(raw_interaction).all():
        raise RuntimeError("source interaction labels contain non-finite values")

    cells = cell_index(data, records)
    residual, cell_sizes = conditional_residuals(raw_interaction, cells)
    residual_mean = residual.mean(dim=0)
    residual_std = residual.std(dim=0, unbiased=False)
    if torch.any(residual_std <= 1e-8):
        raise RuntimeError(
            "at least one conditional interaction residual has near-zero "
            "variance; it cannot provide a meaningful MA target")
    standardized = (residual - residual_mean) / residual_std

    if args.mode == "true":
        partner = list(range(len(data)))
        shuffle_info = {
            "num_cells": len(cells),
            "num_cells_with_block_shuffle": 0,
            "num_rows_with_shuffled_pair": 0,
            "num_singleton_rows_kept": len(data),
        }
    else:
        partner, shuffle_info = deranged_partner_indices(cells, len(data),
                                                           args.seed)
    target_columns = TARGET_COLUMNS[args.auxiliary_targets]
    selected_label = standardized[
        torch.tensor(partner, dtype=torch.long)][:, list(target_columns)]
    feature_names = TARGET_FEATURE_NAMES[args.auxiliary_targets]

    prepared_data, prepared_records = [], []
    for index, item in enumerate(data):
        prepared = dict(item)
        # The ``_emb`` suffix intentionally makes collate_data stack these
        # two-value targets into [batch, 2], rather than concatenate them.
        prepared["mechanism_aux_emb"] = selected_label[index].clone()
        prepared_data.append(prepared)

        record = dict(records[index])
        record["mechanism_auxiliary"] = {
            "training_label_zscore": [
                float(value) for value in selected_label[index]],
            "pairing_mode": args.mode,
            "auxiliary_targets": args.auxiliary_targets,
        }
        prepared_records.append(record)

    atomic_torch_save(prepared_data, output / "processed_data.pkl")
    atomic_write_text(output / "synthetic_data.json",
                      json.dumps(prepared_records, indent=2) + "\n")
    dataset_config = yaml.safe_load(required["dataset"].read_text())
    dataset_config["save_dir"] = str(output)
    atomic_write_text(output / "dataset_config.yaml",
                      yaml.safe_dump(dataset_config, sort_keys=False))

    source_metadata = json.loads(required["metadata"].read_text())
    source_metadata["synthetic_data_json"] = str(output / "synthetic_data.json")
    source_metadata["mechanism_auxiliary"] = {
        "enabled": True,
        "mode": args.mode,
        "auxiliary_targets": args.auxiliary_targets,
        "source_data_dir": str(source),
        "source_feature": "mechanism_emb",
        "training_feature": "mechanism_aux_emb",
        "feature_names": feature_names,
        "feature_transform": "within_cell_center_then_global_zscore",
        "cell_schema": ["sigma_bin", "anion_ratio_bin", "solvent_count_cohort"],
        "sigma_edges_ms_cm": SIGMA_EDGES.tolist(),
        "anion_ratio_edges": RATIO_EDGES.tolist(),
        "residual_mean_before_zscore": [float(value) for value in residual_mean],
        "residual_std_before_zscore": [float(value) for value in residual_std],
        "cell_sizes": cell_sizes,
        "shuffle": shuffle_info,
        "shuffle_seed": args.seed if args.mode == "shuffle_within_cell" else None,
        "pairing_preserved": args.mode == "true",
        "generation_condition": ["conductivity_normalized", "anion_ratio"],
        "inference_mechanism_input": None,
    }
    atomic_write_text(output / "synthetic_metadata.json",
                      json.dumps(source_metadata, indent=2) + "\n")
    # This final sidecar acts as a completion sentinel for MA data reuse.
    atomic_write_text(output / "mechanism_auxiliary.json", json.dumps({
        "mode": args.mode,
        "auxiliary_targets": args.auxiliary_targets,
        "source_data_dir": str(source),
        "output_dir": str(output),
        "num_formulas": len(data),
        "same_formula_order_as_source": True,
        "source_feature": "mechanism_emb",
        "training_feature": "mechanism_aux_emb",
        "mechanism_feature_names": [f"{name}_zscore" for name in feature_names],
        "label_dim": len(target_columns),
        "residual_mean_before_zscore": [float(value) for value in residual_mean],
        "residual_std_before_zscore": [float(value) for value in residual_std],
        "cell_schema": ["sigma_bin", "anion_ratio_bin", "solvent_count_cohort"],
        "shuffle": shuffle_info,
    }, indent=2) + "\n")
    print(
        f"Prepared {args.mode} MA data: {len(data)} formulas -> {output}",
        flush=True)


if __name__ == "__main__":
    main()
