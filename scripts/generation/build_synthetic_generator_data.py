#!/usr/bin/env python3
"""Create matched synthetic formula embeddings for one solvent vocabulary.

Synthetic formulas are sampled with fixed, reproducible composition rules and
scored by the frozen formula oracle.  The resulting serialized FormulaData
contains only the tensors needed by the generator trainer: formula embedding,
BOW, and the frozen oracle's jointly predicted conductivity and anion ratio.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from formula_design.data import Data, FormulaData, collate_data
from formula_design.mol import Molecule
from formula_design.predictor import MolMix, Mono


def load_oracle(config_path, checkpoint):
    config = yaml.safe_load(Path(config_path).read_text())
    pretrain = config["model"]["pretrain"]["model_config"]
    model = Mono(**pretrain)
    model.load_ckpt(str(Path(checkpoint) / "pretrain.pt"))
    formula = MolMix(model, **config["model"]["model"])
    formula.load_ckpt(str(Path(checkpoint) / "optimal.pt"))
    return formula.eval().to("cuda")


def random_formula(solvents, salts, rng, temperature, concentration):
    n_solvents = rng.randint(1, min(3, len(solvents)))
    solvent_ids = rng.sample(range(len(solvents)), n_solvents)
    solvent_weights = np.random.default_rng(rng.randrange(2**32)).dirichlet(
        np.ones(n_solvents)).tolist()
    salt_id = rng.randrange(len(salts))
    solv_names = [solvents[i]["name"] for i in solvent_ids]
    solv_smiles = [Molecule.from_smiles(solvents[i]["smiles"]).get_mapped_smiles()
                   for i in solvent_ids]
    return FormulaData(
        solv_names, solv_smiles,
        [salts[salt_id]["name"]],
        [Molecule.from_smiles(salts[salt_id]["smiles"]).get_mapped_smiles()],
        solvent_weights, [1.0],
        temperature=temperature, concentration=concentration,
        anion_ratio=.5), solvent_ids, salt_id, solvent_weights


def balanced_joint_indices(data, target_count, seed):
    """Round-robin select across oracle conductivity/anion-ratio bins."""
    conductivity_edges = np.asarray([0., 5., 10., 20., 30., 50., 100.])
    ratio_edges = np.asarray([0., .2, .4, .6, .8, 1.])
    buckets = defaultdict(list)
    for index, item in enumerate(data):
        conductivity = float(torch.exp(item["conductivity"] * 8.3344 - 4.6002))
        anion_ratio = float(item["anion_ratio"])
        if not (np.isfinite(conductivity) and 0. <= conductivity <= 100.
                and np.isfinite(anion_ratio) and 0. <= anion_ratio <= 1.):
            continue
        c_bin = min(np.digitize(conductivity, conductivity_edges) - 1,
                    len(conductivity_edges) - 2)
        r_bin = min(np.digitize(anion_ratio, ratio_edges) - 1,
                    len(ratio_edges) - 2)
        buckets[(int(c_bin), int(r_bin))].append(index)

    rng = random.Random(seed)
    keys = list(buckets)
    for values in buckets.values():
        rng.shuffle(values)
    rng.shuffle(keys)
    cursor = {key: 0 for key in keys}
    selected = []
    while len(selected) < target_count:
        made_progress = False
        for key in keys:
            position = cursor[key]
            if position < len(buckets[key]):
                selected.append(buckets[key][position])
                cursor[key] += 1
                made_progress = True
                if len(selected) == target_count:
                    break
        if not made_progress:
            break
    if len(selected) < target_count:
        raise RuntimeError(
            f"only {len(selected)} valid joint-condition samples for requested "
            f"{target_count}; increase --oversample-factor")
    return selected, len(buckets)


def sanitize_legacy_labels(conductivity, anion_ratio):
    """Keep every G1 synthetic formula while making labels trainable.

    This reproduces the practical intent of the earlier unfiltered G1 data:
    an out-of-domain formula is retained instead of being rejected.  Raw
    NaN/Inf values cannot be sent to a diffusion loss, so invalid conductivity
    is assigned Bamboo's historical 10 mS/cm fallback and invalid anion ratio
    is assigned 0.5; finite out-of-range ratios are clipped to [0, 1].
    """
    fallback_conductivity = (math.log(10.0) + 4.6002) / 8.3344
    conductivity_ms_cm = torch.exp(conductivity * 8.3344 - 4.6002)
    valid_conductivity = (
        torch.isfinite(conductivity_ms_cm)
        & (conductivity_ms_cm >= 0.0)
        & (conductivity_ms_cm <= 100.0)
    )
    conductivity = torch.where(
        valid_conductivity,
        conductivity,
        torch.full_like(conductivity, fallback_conductivity),
    )
    anion_ratio = torch.nan_to_num(anion_ratio, nan=0.5, posinf=0.5,
                                   neginf=0.5).clamp_(0.0, 1.0)
    return conductivity, anion_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvents", required=True)
    parser.add_argument("--salts", required=True)
    parser.add_argument("--oracle-config", required=True)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-formulas", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--concentration", type=float, default=0.10)
    parser.add_argument("--oversample-factor", type=float, default=1.5,
                        help="Generate extra random formulas before balanced "
                        "joint-condition selection.")
    parser.add_argument("--label-policy", choices=("strict", "sanitize_keep_all"),
                        default="strict",
                        help="strict rejects out-of-domain joint labels; "
                        "sanitize_keep_all retains every formula after "
                        "numerical fallback/clipping for a legacy-G1 control.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    solvents = json.loads(Path(args.solvents).read_text())
    salts = json.loads(Path(args.salts).read_text())
    rng = random.Random(args.seed)
    temperature = (args.temperature_c + 273.15) / 373.15
    candidate_count = int(np.ceil(args.num_formulas * args.oversample_factor))
    print(
        "Preparing %d oracle-scored candidate formulas in batches of %d "
        "for joint conductivity/anion-ratio balancing."
        % (candidate_count, args.batch_size),
        flush=True,
    )
    oracle = load_oracle(args.oracle_config, args.oracle_checkpoint)
    generator_data = []
    with torch.no_grad():
        # Do not materialize every FormulaData graph at once.  With joint
        # condition balancing the default candidate count is 45,000; holding
        # all RDKit/DGL graph objects until oracle inference can exhaust host
        # memory before the first progress message is printed.
        for start in range(0, candidate_count, args.batch_size):
            current_size = min(args.batch_size, candidate_count - start)
            formula_data, formula_records = [], []
            for _ in range(current_size):
                data, solvent_ids, salt_id, solvent_weights = random_formula(
                    solvents, salts, rng, temperature, args.concentration)
                formula_data.append(data)
                formula_records.append((solvent_ids, salt_id, solvent_weights))

            batch = collate_data(formula_data)
            prediction = oracle(batch.to("cuda"))
            h_inv = prediction["h_inv"].detach().cpu()
            conductivity = prediction["conductivity"].detach().cpu().reshape(-1)
            anion_ratio = prediction["anion_ratio"].detach().cpu().reshape(-1)
            if args.label_policy == "sanitize_keep_all":
                conductivity, anion_ratio = sanitize_legacy_labels(
                    conductivity, anion_ratio)
            for index in range(current_size):
                solvent_ids, salt_id, solvent_weights = formula_records[index]
                bow = torch.zeros(len(solvents) + len(salts), dtype=torch.float32)
                bow[solvent_ids] = torch.tensor(solvent_weights, dtype=torch.float32)
                bow[len(solvents) + salt_id] = 1.0
                item = Data()
                item.frm_emb = h_inv[index]
                item.bow_vec = bow
                item.conductivity = conductivity[index].reshape(1)
                # The condition must describe this formula, rather than a
                # globally fixed requested value.  This makes the diffusion
                # model learn the joint (conductivity, anion_ratio) manifold.
                item.anion_ratio = anion_ratio[index].reshape(1)
                generator_data.append(dict(item))
            print(f"oracle synthetic labels: {start + current_size}/{candidate_count}")

    if args.label_policy == "strict":
        selected_indices, nonempty_bins = balanced_joint_indices(
            generator_data, args.num_formulas, args.seed)
    else:
        selected_indices = list(range(len(generator_data)))
        nonempty_bins = None
    generator_data = [generator_data[index] for index in selected_indices]

    torch.save(generator_data, output / "processed_data.pkl")
    (output / "dataset_config.yaml").write_text(yaml.safe_dump({
        "data_cls": "FormulaData", "save_dir": str(output),
        "json_path": "", "key_map": {}, "kwargs": {},
    }, sort_keys=False))
    (output / "synthetic_metadata.json").write_text(json.dumps({
        "num_formulas": args.num_formulas,
        "candidate_formulas_before_selection": candidate_count,
        "joint_condition_label_source": "frozen_oracle: conductivity + anion_ratio",
        "joint_condition_sampling": (
            "round_robin_2d_bins" if args.label_policy == "strict"
            else "all_candidates_after_label_sanitization"
        ),
        "label_policy": args.label_policy,
        "nonempty_joint_bins": nonempty_bins,
        "num_solvents": len(solvents),
        "num_salts": len(salts), "temperature_c": args.temperature_c,
        "concentration": args.concentration,
        "oversample_factor": args.oversample_factor,
        "seed": args.seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
