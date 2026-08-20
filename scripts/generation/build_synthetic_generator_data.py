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
from collections import Counter, defaultdict
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


_BAMBOO_SOLVENT_COUNT_WEIGHTS = {
    # Exact frequencies in Bamboo-Mixer's released 30,000-row
    # run/synthetic_data.json.  Its training regime is mostly 4--6 solvents.
    1: 62, 2: 1891, 3: 2778, 4: 5723, 5: 8406, 6: 11140,
}


def bamboo_count_schedule(total_count, max_solvents, seed):
    """Return a shuffled, quota-exact rescaling of Bamboo's count profile."""
    counts = [count for count in _BAMBOO_SOLVENT_COUNT_WEIGHTS
              if count <= max_solvents]
    weights = [_BAMBOO_SOLVENT_COUNT_WEIGHTS[count] for count in counts]
    weight_sum = sum(weights)
    raw = [total_count * weight / weight_sum for weight in weights]
    quotas = [int(value) for value in raw]
    for index in sorted(range(len(counts)),
                        key=lambda i: (raw[i] - quotas[i], counts[i]),
                        reverse=True)[:total_count - sum(quotas)]:
        quotas[index] += 1
    schedule = [count for count, quota in zip(counts, quotas)
                for _ in range(quota)]
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def random_formula(solvents, salts, rng, temperature, concentration,
                   solvent_count_policy, salt_mixture_mode, salt_mixture_bins,
                   n_solvents=None):
    """Sample a formula with normalized solvent and salt sub-compositions.

    Bamboo represents solvent ratios and salt ratios as two independently
    normalized parts of the formula BOW.  Retaining a single salt at ratio one
    would collapse the latter manifold and prevents the controlled generator
    from learning mixed-salt/anionic-ratio behaviour.
    """
    if n_solvents is not None:
        pass
    elif solvent_count_policy == "bamboo_empirical_1_to_6":
        counts = [count for count in _BAMBOO_SOLVENT_COUNT_WEIGHTS
                  if count <= len(solvents)]
        weights = [_BAMBOO_SOLVENT_COUNT_WEIGHTS[count] for count in counts]
        n_solvents = rng.choices(counts, weights=weights, k=1)[0]
    else:
        n_solvents = rng.randint(1, min(3, len(solvents)))
    solvent_ids = rng.sample(range(len(solvents)), n_solvents)
    solvent_weights = np.random.default_rng(rng.randrange(2**32)).dirichlet(
        np.ones(n_solvents)).tolist()
    salt_mixture_bin = None
    if salt_mixture_mode in ("stratified_two_salt", "bamboo_fixed_two_salt"):
        if len(salts) < 2:
            raise ValueError(f"{salt_mixture_mode} requires at least two salts")
        n_salts = 2
    else:
        n_salts = rng.randint(1, min(2, len(salts)))
    salt_ids = (list(range(2)) if salt_mixture_mode == "bamboo_fixed_two_salt"
                else rng.sample(range(len(salts)), n_salts))
    if salt_mixture_mode == "stratified_two_salt":
        # This is an *upstream composition* stratification, not an artificial
        # anion-ratio label.  The condition label below remains the frozen
        # oracle prediction for the sampled formula, so it stays consistent
        # with the generator's latent target.
        salt_mixture_bin = rng.randrange(salt_mixture_bins)
        lower = salt_mixture_bin / salt_mixture_bins
        upper = (salt_mixture_bin + 1) / salt_mixture_bins
        first_weight = lower + (upper - lower) * rng.random()
        salt_weights = [first_weight, 1.0 - first_weight]
    else:
        salt_weights = np.random.default_rng(rng.randrange(2**32)).dirichlet(
            np.ones(n_salts)).tolist()
    solv_names = [solvents[i]["name"] for i in solvent_ids]
    solv_smiles = [Molecule.from_smiles(solvents[i]["smiles"]).get_mapped_smiles()
                   for i in solvent_ids]
    return FormulaData(
        solv_names, solv_smiles,
        [salts[i]["name"] for i in salt_ids],
        [Molecule.from_smiles(salts[i]["smiles"]).get_mapped_smiles()
         for i in salt_ids],
        solvent_weights, salt_weights,
        temperature=temperature, concentration=concentration,
        # Filled with the oracle prediction after this formula is scored.
        anion_ratio=.5), solvent_ids, salt_ids, solvent_weights, salt_weights, salt_mixture_bin


def balanced_joint_indices(data, target_count, seed, strategy,
                           min_nonempty_anion_bins):
    """Select samples with anion-first, then conductivity, balancing.

    A flat joint-bin round robin gives more total samples to an anion-ratio
    region whenever that region happens to occupy more conductivity bins.
    The hierarchical strategy fixes that: it visits nonempty anion bins in
    turn, then rotates through conductivity bins within each anion bin.
    """
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
    observed_anion_bins = {r_bin for _, r_bin in buckets}
    if (strategy != "none"
            and len(observed_anion_bins) < min_nonempty_anion_bins):
        raise RuntimeError(
            "only %d nonempty oracle anion-ratio bins; require at least %d. "
            "Increase composition coverage or relax --min-nonempty-anion-bins; "
            "do not train a conditional diffusion model on unsupported labels."
            % (len(observed_anion_bins), min_nonempty_anion_bins))
    selected = []
    if strategy == "none":
        # Keep valid labels but preserve their natural, un-reweighted joint
        # distribution, matching the absence of bin balancing in Bamboo.
        selected = [index for values in buckets.values() for index in values]
        selected.sort()
        selected = selected[:target_count]
    elif strategy == "flat_round_robin":
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
    else:
        by_ratio = defaultdict(list)
        for c_bin, r_bin in keys:
            by_ratio[r_bin].append((c_bin, r_bin))
        ratio_bins = list(by_ratio)
        rng.shuffle(ratio_bins)
        inner_cursor = {r_bin: 0 for r_bin in ratio_bins}
        for r_bin in ratio_bins:
            rng.shuffle(by_ratio[r_bin])
        while len(selected) < target_count:
            made_progress = False
            for r_bin in ratio_bins:
                group = by_ratio[r_bin]
                for _ in range(len(group)):
                    key = group[inner_cursor[r_bin] % len(group)]
                    inner_cursor[r_bin] += 1
                    position = cursor[key]
                    if position < len(buckets[key]):
                        selected.append(buckets[key][position])
                        cursor[key] += 1
                        made_progress = True
                        break
                if len(selected) == target_count:
                    break
            if not made_progress:
                break
    if len(selected) < target_count:
        raise RuntimeError(
            f"only {len(selected)} valid joint-condition samples for requested "
            f"{target_count}; increase --oversample-factor")
    def count_bins(indices):
        result = Counter()
        for index in indices:
            conductivity = float(torch.exp(data[index]["conductivity"] * 8.3344 - 4.6002))
            anion_ratio = float(data[index]["anion_ratio"])
            c_bin = min(np.digitize(conductivity, conductivity_edges) - 1,
                        len(conductivity_edges) - 2)
            r_bin = min(np.digitize(anion_ratio, ratio_edges) - 1,
                        len(ratio_edges) - 2)
            result[f"conductivity_{int(c_bin)}__anion_{int(r_bin)}"] += 1
        return dict(sorted(result.items()))

    def ratio_counts(indices):
        result = Counter()
        for index in indices:
            anion_ratio = float(data[index]["anion_ratio"])
            r_bin = min(np.digitize(anion_ratio, ratio_edges) - 1,
                        len(ratio_edges) - 2)
            result[f"anion_{int(r_bin)}"] += 1
        return dict(sorted(result.items()))

    balance_info = {
        "strategy": strategy,
        "conductivity_edges_ms_cm": conductivity_edges.tolist(),
        "anion_ratio_edges": ratio_edges.tolist(),
        "nonempty_joint_bins": len(buckets),
        "nonempty_anion_bins": len({r_bin for _, r_bin in buckets}),
        "anion_bin_counts_before_selection": ratio_counts(
            [index for values in buckets.values() for index in values]),
        "anion_bin_counts_after_selection": ratio_counts(selected),
        "joint_bin_counts_before_selection": count_bins(
            [index for values in buckets.values() for index in values]),
        "joint_bin_counts_after_selection": count_bins(selected),
    }
    return selected, balance_info


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
    parser.add_argument("--salt-mixture-mode",
                        choices=("random_1_to_2_salt", "stratified_two_salt",
                                 "bamboo_fixed_two_salt"),
                        default="random_1_to_2_salt",
                        help="random_1_to_2_salt reproduces the regular synthetic "
                        "sampler. stratified_two_salt samples two salts with their "
                        "first-salt fraction uniformly across mixture bins before "
                        "the frozen oracle supplies the actual condition label.")
    parser.add_argument("--salt-mixture-bins", type=int, default=5)
    parser.add_argument("--solvent-count-policy",
                        choices=("uniform_1_to_3", "bamboo_empirical_1_to_6"),
                        default="uniform_1_to_3",
                        help="bamboo_empirical_1_to_6 samples 1--6 solvents "
                        "using the released Bamboo synthetic-data frequencies.")
    parser.add_argument("--joint-balance-strategy",
                        choices=("none", "flat_round_robin",
                                 "hierarchical_anion_then_conductivity"),
                        default="hierarchical_anion_then_conductivity")
    parser.add_argument("--min-nonempty-anion-bins", type=int, default=1,
                        help="Fail rather than train if the frozen-oracle labels "
                        "do not cover this many anion-ratio bins.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.salt_mixture_bins < 2:
        parser.error("--salt-mixture-bins must be at least 2")
    if not 1 <= args.min_nonempty_anion_bins <= 5:
        parser.error("--min-nonempty-anion-bins must be in [1, 5]")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    solvents = json.loads(Path(args.solvents).read_text())
    salts = json.loads(Path(args.salts).read_text())
    rng = random.Random(args.seed)
    temperature = (args.temperature_c + 273.15) / 373.15
    candidate_count = int(np.ceil(args.num_formulas * args.oversample_factor))
    if args.solvent_count_policy == "bamboo_empirical_1_to_6":
        solvent_count_schedule = bamboo_count_schedule(
            candidate_count, len(solvents), args.seed)
    else:
        solvent_count_schedule = None
    print(
        "Preparing %d oracle-scored candidate formulas in batches of %d "
        "for joint conductivity/anion-ratio balancing."
        % (candidate_count, args.batch_size),
        flush=True,
    )
    oracle = load_oracle(args.oracle_config, args.oracle_checkpoint)
    generator_data = []
    synthetic_formula_records = []
    source_salt_mixture_bin_counts = Counter()
    with torch.no_grad():
        # Do not materialize every FormulaData graph at once.  With joint
        # condition balancing the default candidate count is 45,000; holding
        # all RDKit/DGL graph objects until oracle inference can exhaust host
        # memory before the first progress message is printed.
        for start in range(0, candidate_count, args.batch_size):
            current_size = min(args.batch_size, candidate_count - start)
            formula_data, formula_records = [], []
            for offset in range(current_size):
                global_index = start + offset
                (data, solvent_ids, salt_ids, solvent_weights, salt_weights,
                 salt_mixture_bin) = random_formula(
                    solvents, salts, rng, temperature, args.concentration,
                    args.solvent_count_policy,
                    args.salt_mixture_mode, args.salt_mixture_bins,
                    (solvent_count_schedule[global_index]
                     if solvent_count_schedule is not None else None))
                formula_data.append(data)
                formula_records.append((solvent_ids, salt_ids, solvent_weights,
                                        salt_weights))
                if salt_mixture_bin is not None:
                    source_salt_mixture_bin_counts[str(salt_mixture_bin)] += 1

            batch = collate_data(formula_data)
            prediction = oracle(batch.to("cuda"))
            h_inv = prediction["h_inv"].detach().cpu()
            conductivity = prediction["conductivity"].detach().cpu().reshape(-1)
            anion_ratio = prediction["anion_ratio"].detach().cpu().reshape(-1)
            if args.label_policy == "sanitize_keep_all":
                conductivity, anion_ratio = sanitize_legacy_labels(
                    conductivity, anion_ratio)
            for index in range(current_size):
                solvent_ids, salt_ids, solvent_weights, salt_weights = formula_records[index]
                bow = torch.zeros(len(solvents) + len(salts), dtype=torch.float32)
                bow[solvent_ids] = torch.tensor(solvent_weights, dtype=torch.float32)
                bow[[len(solvents) + salt_id for salt_id in salt_ids]] = (
                    torch.tensor(salt_weights, dtype=torch.float32))
                item = Data()
                item.frm_emb = h_inv[index]
                item.bow_vec = bow
                item.conductivity = conductivity[index].reshape(1)
                # The condition must describe this formula, rather than a
                # globally fixed requested value.  This makes the diffusion
                # model learn the joint (conductivity, anion_ratio) manifold.
                item.anion_ratio = anion_ratio[index].reshape(1)
                generator_data.append(dict(item))
                conductivity_ms_cm = float(torch.exp(
                    conductivity[index] * 8.3344 - 4.6002))
                synthetic_formula_records.append({
                    "solvents": [{"name": solvents[solvent_id]["name"],
                                  "smiles": solvents[solvent_id]["smiles"],
                                  "molar_ratio": float(weight)}
                                 for solvent_id, weight in zip(solvent_ids, solvent_weights)],
                    "salts": [{"name": salts[salt_id]["name"],
                               "smiles": salts[salt_id]["smiles"],
                               "molar_ratio": float(weight)}
                              for salt_id, weight in zip(salt_ids, salt_weights)],
                    "temperature": args.temperature_c,
                    "salt_molar_ratio": args.concentration,
                    "conductivity": conductivity_ms_cm,
                    "anion_ratio": float(anion_ratio[index]),
                    "label_source": "frozen_oracle",
                })
            print(f"oracle synthetic labels: {start + current_size}/{candidate_count}")

    if args.label_policy == "strict":
        selected_indices, balance_info = balanced_joint_indices(
            generator_data, args.num_formulas, args.seed,
            args.joint_balance_strategy, args.min_nonempty_anion_bins)
    else:
        selected_indices = list(range(len(generator_data)))
        balance_info = None
    generator_data = [generator_data[index] for index in selected_indices]
    synthetic_formula_records = [synthetic_formula_records[index]
                                 for index in selected_indices]

    torch.save(generator_data, output / "processed_data.pkl")
    (output / "synthetic_data.json").write_text(
        json.dumps(synthetic_formula_records, indent=2) + "\n")
    (output / "dataset_config.yaml").write_text(yaml.safe_dump({
        "data_cls": "FormulaData", "save_dir": str(output),
        "json_path": "", "key_map": {}, "kwargs": {},
    }, sort_keys=False))
    (output / "synthetic_metadata.json").write_text(json.dumps({
        "num_formulas": args.num_formulas,
        "candidate_formulas_before_selection": candidate_count,
        "conductivity_label_source": "frozen_oracle",
        "anion_ratio_label_source": "frozen_oracle",
        "synthetic_data_json": str(output / "synthetic_data.json"),
        "joint_condition_sampling": (
            args.joint_balance_strategy if args.label_policy == "strict"
            else "all_candidates_after_label_sanitization"
        ),
        "label_policy": args.label_policy,
        "joint_condition_balance": balance_info,
        "min_nonempty_anion_bins_required": args.min_nonempty_anion_bins,
        "num_solvents": len(solvents),
        "solvent_count_policy": args.solvent_count_policy,
        "sampled_solvent_count_counts": (
            dict(sorted(Counter(solvent_count_schedule).items()))
            if solvent_count_schedule is not None else None),
        "num_salts": len(salts),
        "salt_sampling": args.salt_mixture_mode,
        "salt_mixture_bins": args.salt_mixture_bins,
        "source_salt_mixture_bin_counts": dict(source_salt_mixture_bin_counts),
        "temperature_c": args.temperature_c,
        "concentration": args.concentration,
        "oversample_factor": args.oversample_factor,
        "seed": args.seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
