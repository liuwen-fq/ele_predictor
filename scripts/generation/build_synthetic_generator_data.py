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
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from formula_design.data import Data, FormulaData, collate_data
from formula_design.mol import Molecule
from formula_design.predictor.oracle_adapter import load_bamboo_oracle as load_oracle


_BAMBOO_SOLVENT_COUNT_WEIGHTS = {
    # Exact frequencies in Bamboo-Mixer's released 30,000-row
    # run/synthetic_data.json.  Its training regime is mostly 4--6 solvents.
    1: 62, 2: 1891, 3: 2778, 4: 5723, 5: 8406, 6: 11140,
}


def atomic_torch_save(value, path):
    """Publish a torch artifact only after its complete payload is written."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path, text):
    """Atomically publish a small UTF-8 sidecar file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


# These bins are used only to preserve the *existing* synthetic-data
# composition while choosing which candidates enter a matched control or an
# interaction-coverage training set.  They deliberately match the reporting
# bins around common target values, but they are not target-specific weights.
_MECHANISM_SIGMA_EDGES = np.asarray([0., 5., 10., 15., 20., 25., 30.,
                                     40., 100.])
_MECHANISM_RATIO_EDGES = np.asarray([0., .2, .4, .6, .8, 1.])


def explanation_interaction_terms(prediction, batch_size):
    """Return formula-level interaction proxies from the explanation head.

    The values remain model-internal normalized contributions: they are saved
    for mechanism coverage and future RDF-compatible metadata, never used as
    physical interaction energies or as direct property labels.
    """
    explanation = prediction.get("property_fragment_explanations")
    if explanation is None:
        diagnostics = prediction.get("fragment_diagnostics", {})
        explanation = (diagnostics.get("property_explanations")
                       or diagnostics.get("explanations"))
    if explanation is None or "tasks" not in explanation:
        raise RuntimeError(
            "--mechanism-selection requires a property/fragment explanation "
            "head, but the frozen oracle returned no explanation tasks")
    tasks = explanation["tasks"]
    required = ("conductivity_explain", "anion_ratio_explain")
    missing = [name for name in required if name not in tasks]
    if missing:
        raise RuntimeError(
            "--mechanism-selection requires explanation tasks %s; missing %s"
            % (", ".join(required), ", ".join(missing)))

    def get_task_term(name):
        value = tasks[name].get("interaction_sum")
        if value is None:
            raise RuntimeError(f"explanation task {name} has no interaction_sum")
        value = value.detach().cpu().reshape(-1)
        if value.numel() != batch_size:
            raise RuntimeError(
                f"explanation task {name} returned {value.numel()} terms for "
                f"a batch of {batch_size}")
        return value

    return get_task_term("conductivity_explain"), get_task_term("anion_ratio_explain")


def mechanism_cell(item, record):
    """Joint-label plus solvent-count cell used for matched selection."""
    sigma = float(torch.exp(item["conductivity"] * 8.3344 - 4.6002))
    ratio = float(item["anion_ratio"])
    sigma_bin = min(np.digitize(sigma, _MECHANISM_SIGMA_EDGES) - 1,
                    len(_MECHANISM_SIGMA_EDGES) - 2)
    ratio_bin = min(np.digitize(ratio, _MECHANISM_RATIO_EDGES) - 1,
                    len(_MECHANISM_RATIO_EDGES) - 2)
    n_solvents = len(record["solvents"])
    solvent_cohort = "1" if n_solvents == 1 else ("2_3" if n_solvents <= 3 else "4_6")
    return int(sigma_bin), int(ratio_bin), solvent_cohort


def matched_mechanism_indices(data, records, target_count, seed, mode):
    """Choose a matched random control or interaction-coverage subset.

    Both modes first define the same seeded random reference subset and its
    exact (sigma-bin, ratio-bin, solvent-count-cohort) quotas.  The
    interaction mode then changes only *within-cell* sampling.  Consequently
    Q/I have the same final sample count and label/composition marginals.
    """
    if len(data) < target_count:
        raise RuntimeError(
            f"mechanism selection needs {target_count} candidates, got {len(data)}; "
            "increase --oversample-factor")
    if len(data) != len(records):
        raise RuntimeError("generator data and synthetic records are misaligned")

    base_rng = random.Random(seed)
    random_reference = base_rng.sample(range(len(data)), target_count)
    quotas = Counter(mechanism_cell(data[index], records[index])
                     for index in random_reference)
    if mode == "matched_random":
        return random_reference, {
            "mode": mode,
            "selection_seed": seed,
            "quota_cells": {"|".join(map(str, cell)): count
                            for cell, count in sorted(quotas.items())},
            "interaction_coverage": "not_applied",
        }

    cells = defaultdict(list)
    for index in range(len(data)):
        cells[mechanism_cell(data[index], records[index])].append(index)
    select_rng = random.Random(seed + 104729)
    selected, strata_info = [], {}
    for cell in sorted(quotas):
        required = quotas[cell]
        candidates = list(cells[cell])
        if len(candidates) < required:
            raise RuntimeError(
                f"cell {cell} has {len(candidates)} candidates for quota {required}")
        # Single-solvent formulas have no cross-molecule pair contribution.
        # Preserve their matched-random composition rather than inventing a
        # meaningless interaction stratum.
        if cell[2] == "1":
            select_rng.shuffle(candidates)
            selected.extend(candidates[:required])
            strata_info["|".join(map(str, cell))] = {
                "quota": required, "strategy": "single_solvent_random",
            }
            continue

        values = np.asarray([
            float(data[index]["mechanism_emb"][0]) for index in candidates
        ], dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 3:
            select_rng.shuffle(candidates)
            selected.extend(candidates[:required])
            strata_info["|".join(map(str, cell))] = {
                "quota": required, "strategy": "insufficient_finite_interactions",
            }
            continue
        lower, upper = np.quantile(values[finite], [1. / 3., 2. / 3.])
        strata = [[], [], []]
        for index, value in zip(candidates, values):
            if not np.isfinite(value):
                strata[1].append(index)
            elif value <= lower:
                strata[0].append(index)
            elif value <= upper:
                strata[1].append(index)
            else:
                strata[2].append(index)
        for values_in_stratum in strata:
            select_rng.shuffle(values_in_stratum)
        desired = [required // 3] * 3
        for index in range(required % 3):
            desired[index] += 1
        chosen = []
        for values_in_stratum, count in zip(strata, desired):
            chosen.extend(values_in_stratum[:count])
        # Equal thirds can be impossible because quantile ties or a sparse
        # tail.  Fill only from unused candidates in the same matched cell;
        # never duplicate formulas or alter its label quota.
        if len(chosen) < required:
            chosen_set = set(chosen)
            remainder = [index for index in candidates if index not in chosen_set]
            select_rng.shuffle(remainder)
            chosen.extend(remainder[:required - len(chosen)])
        if len(chosen) != required:
            raise RuntimeError(f"could not fill interaction quota for cell {cell}")
        selected.extend(chosen)
        strata_info["|".join(map(str, cell))] = {
            "quota": required,
            "strategy": "sigma_interaction_tertiles",
            "sigma_interaction_quantiles": [float(lower), float(upper)],
            "available_by_stratum": [len(values) for values in strata],
            "selected_by_stratum": [sum(index in set(chosen) for index in values)
                                    for values in strata],
        }

    if len(selected) != target_count or len(set(selected)) != target_count:
        raise RuntimeError("interaction selection did not produce a unique matched subset")
    select_rng.shuffle(selected)
    return selected, {
        "mode": mode,
        "selection_seed": seed,
        "quota_cells": {"|".join(map(str, cell)): count
                        for cell, count in sorted(quotas.items())},
        "interaction_coverage": "conductivity_explain.interaction_sum tertiles",
        "cell_strata": strata_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvents", required=True)
    parser.add_argument("--salts", required=True)
    parser.add_argument("--oracle-config", required=True)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-cache-dir",
                        help="Optional shared, oracle-scored candidate cache. "
                        "Paired Q/I runs must point to the same cache so they "
                        "select from exactly the same oversampled formulas.")
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
    parser.add_argument("--mechanism-selection",
                        choices=("none", "matched_random",
                                 "interaction_sigma_tertiles"),
                        default="none",
                        help="For matched Q/I experiments, retain the same "
                        "joint-label/solvent-count quotas while selecting "
                        "either a seeded random control or sigma-interaction "
                        "coverage within each cell.  This does not change "
                        "diffusion conditions.")
    parser.add_argument("--min-nonempty-anion-bins", type=int, default=1,
                        help="Fail rather than train if the frozen-oracle labels "
                        "do not cover this many anion-ratio bins.")
    parser.add_argument("--anion-ratio-min", type=float, default=0.0,
                        help="inclusive frozen-oracle anion-ratio eligibility "
                        "lower bound for synthetic generator data")
    parser.add_argument("--anion-ratio-max", type=float, default=1.0,
                        help="inclusive frozen-oracle anion-ratio eligibility "
                        "upper bound for synthetic generator data")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.salt_mixture_bins < 2:
        parser.error("--salt-mixture-bins must be at least 2")
    if not 1 <= args.min_nonempty_anion_bins <= 5:
        parser.error("--min-nonempty-anion-bins must be in [1, 5]")
    if not (0.0 <= args.anion_ratio_min < args.anion_ratio_max <= 1.0):
        parser.error("require 0 <= --anion-ratio-min < --anion-ratio-max <= 1")

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
    generator_data = []
    synthetic_formula_records = []
    source_salt_mixture_bin_counts = Counter()
    cache_path = None
    if args.candidate_cache_dir:
        cache_dir = Path(args.candidate_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "oracle_scored_candidates.pt"
    if cache_path is not None and cache_path.is_file():
        cached = torch.load(cache_path, weights_only=False)
        cache_meta = cached.get("metadata", {})
        expected = {
            "candidate_count": candidate_count,
            "seed": args.seed,
            "num_solvents": len(solvents),
            "num_salts": len(salts),
            "solvent_count_policy": args.solvent_count_policy,
            "salt_mixture_mode": args.salt_mixture_mode,
        }
        mismatch = {key: (cache_meta.get(key), value)
                    for key, value in expected.items()
                    if cache_meta.get(key) != value}
        if mismatch:
            raise RuntimeError(
                "shared candidate cache does not match this Q/I run: "
                + ", ".join(f"{key}={old!r} (expected {new!r})"
                            for key, (old, new) in mismatch.items()))
        generator_data = cached["generator_data"]
        synthetic_formula_records = cached["synthetic_formula_records"]
        source_salt_mixture_bin_counts = Counter(
            cached.get("source_salt_mixture_bin_counts", {}))
        if args.mechanism_selection != "none" and (
                not generator_data or "mechanism_emb" not in generator_data[0]):
            raise RuntimeError(
                "shared candidate cache lacks interaction proxies; remove this "
                "cache or rebuild it with --mechanism-selection")
        print(f"Reusing shared oracle-scored candidate cache: {cache_path}", flush=True)
    else:
        oracle = load_oracle(args.oracle_config, args.oracle_checkpoint)
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
                if args.mechanism_selection != "none":
                    interaction_sigma, interaction_anion = explanation_interaction_terms(
                        prediction, current_size)
                else:
                    interaction_sigma = interaction_anion = None
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
                    if interaction_sigma is not None:
                        # Generic two-value mechanism interface.  It is ignored
                        # by the current two-condition diffusion/decoder, but it
                        # makes the training data auditable and can later be
                        # replaced by an RDF mechanism embedding without changing
                        # the formula-data contract.
                        item.mechanism_emb = torch.stack((
                            interaction_sigma[index], interaction_anion[index]))
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
                        **({
                            "mechanism_proxy": {
                                "source": "explanation_head_v1",
                                "conductivity_interaction_sum_normalized": float(
                                    interaction_sigma[index]),
                                "anion_ratio_interaction_sum_normalized": float(
                                    interaction_anion[index]),
                            },
                        } if interaction_sigma is not None else {}),
                    })
                print(f"oracle synthetic labels: {start + current_size}/{candidate_count}")
        if cache_path is not None:
            torch.save({
                "generator_data": generator_data,
                "synthetic_formula_records": synthetic_formula_records,
                "source_salt_mixture_bin_counts": dict(source_salt_mixture_bin_counts),
                "metadata": {
                    "candidate_count": candidate_count,
                    "seed": args.seed,
                    "num_solvents": len(solvents),
                    "num_salts": len(salts),
                    "solvent_count_policy": args.solvent_count_policy,
                    "salt_mixture_mode": args.salt_mixture_mode,
                },
            }, cache_path)
            print(f"Wrote shared oracle-scored candidate cache: {cache_path}", flush=True)

    anion_window_candidate_count = len(generator_data)
    anion_window_accepted_count = anion_window_candidate_count
    if args.label_policy == "strict":
        # Anion ratio is a *formula*-level label.  Apply its common 0.12--0.68
        # eligibility interval only after the complete salt/solvent formula
        # has been scored by the frozen oracle; this avoids assigning a false
        # single-solvent transport label during vocabulary construction.
        kept = [index for index, item in enumerate(generator_data)
                if (args.anion_ratio_min <= float(item["anion_ratio"])
                    <= args.anion_ratio_max)]
        generator_data = [generator_data[index] for index in kept]
        synthetic_formula_records = [synthetic_formula_records[index]
                                     for index in kept]
        anion_window_accepted_count = len(generator_data)
        if len(generator_data) < args.num_formulas:
            raise RuntimeError(
                "only %d/%d oracle-scored formulas satisfy anion-ratio window "
                "[%.3f, %.3f]; raise --oversample-factor or relax the window"
                % (len(generator_data), args.num_formulas,
                   args.anion_ratio_min, args.anion_ratio_max))
        selected_indices, balance_info = balanced_joint_indices(
            generator_data, args.num_formulas, args.seed,
            args.joint_balance_strategy, args.min_nonempty_anion_bins)
        if len(selected_indices) != args.num_formulas:
            raise RuntimeError(
                "joint balancing selected %d rather than requested %d formulas; "
                "increase --oversample-factor or relax label constraints"
                % (len(selected_indices), args.num_formulas))
    else:
        selected_indices = list(range(len(generator_data)))
        balance_info = None
    if args.mechanism_selection != "none":
        # The mechanism selector intentionally runs after any strict
        # eligibility filter.  Q and I therefore see the same valid candidate
        # population and have exactly matched label/composition quotas.
        selected_indices, mechanism_selection_info = matched_mechanism_indices(
            generator_data, synthetic_formula_records, args.num_formulas,
            args.seed, args.mechanism_selection)
    else:
        mechanism_selection_info = None
    generator_data = [generator_data[index] for index in selected_indices]
    synthetic_formula_records = [synthetic_formula_records[index]
                                 for index in selected_indices]

    # Publish the required files atomically.  Metadata is written last, so a
    # reader which requires all four files never observes a partial dataset.
    atomic_torch_save(generator_data, output / "processed_data.pkl")
    atomic_write_text(output / "synthetic_data.json",
                      json.dumps(synthetic_formula_records, indent=2) + "\n")
    atomic_write_text(output / "dataset_config.yaml", yaml.safe_dump({
        "data_cls": "FormulaData", "save_dir": str(output),
        "json_path": "", "key_map": {}, "kwargs": {},
    }, sort_keys=False))
    atomic_write_text(output / "synthetic_metadata.json", json.dumps({
        "num_formulas": args.num_formulas,
        "candidate_formulas_before_selection": candidate_count,
        "oracle_scored_formula_count": anion_window_candidate_count,
        "anion_ratio_eligibility_window": {
            "min": args.anion_ratio_min, "max": args.anion_ratio_max,
            "accepted_before_balancing": anion_window_accepted_count,
            "rejected_before_balancing": anion_window_candidate_count - anion_window_accepted_count,
        },
        "conductivity_label_source": "frozen_oracle",
        "anion_ratio_label_source": "frozen_oracle",
        "synthetic_data_json": str(output / "synthetic_data.json"),
        "joint_condition_sampling": (
            args.joint_balance_strategy if args.label_policy == "strict"
            else "all_candidates_after_label_sanitization"
        ),
        "label_policy": args.label_policy,
        "joint_condition_balance": balance_info,
        "mechanism_selection": mechanism_selection_info,
        "mechanism_schema": ({
            "field": "mechanism_emb",
            "source": "explanation_head_v1",
            "features": [
                "conductivity_explain.interaction_sum",
                "anion_ratio_explain.interaction_sum",
            ],
            "selection_feature": "conductivity_explain.interaction_sum",
            "interpretation": "model-internal normalized interaction proxy; not a physical energy",
        } if args.mechanism_selection != "none" else None),
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
