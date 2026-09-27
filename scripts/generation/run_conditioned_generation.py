#!/usr/bin/env python3
"""Sample one trained generator at a target conductivity and score formulas."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from formula_design.data import FormulaData, collate_data
from formula_design.generator import FormulaDecoder, FormulaDiffusion
from formula_design.mol import Molecule
from formula_design.predictor.oracle_adapter import load_bamboo_oracle as load_oracle


def load_generator(config_path, checkpoint, model_cls, model_key):
    config = yaml.safe_load(Path(config_path).read_text())
    # Mechanistic experiments write compact generator-only configs under
    # ``model``.  The released Bamboo checkpoint instead uses its historical
    # ``diffusion_config/decoder_config.model_config`` layout.  Supporting
    # both formats lets G0 be the actual released Bamboo generator rather
    # than a controlled re-training with the same vocabulary.
    if "model" in config and model_key in config["model"]:
        model_config = config["model"][model_key]
    elif model_key == "diffusion_model_config":
        model_config = config["diffusion_config"]["model_config"]
    elif model_key == "decoder_model_config":
        model_config = config["decoder_config"]["model_config"]
    else:
        raise KeyError(f"cannot locate {model_key} in {config_path}")
    model = model_cls(**model_config)
    model.load_ckpt(checkpoint)
    return model.eval().to("cuda")


def load_mechanism_prior(path):
    """Load the empirical p(I_sigma, I_r | sigma, r) sampling table.

    The table is constructed from the training synthetic set before a possible
    M-shuffle perturbation.  It provides a requested-condition mechanism
    sample at generation time; it never evaluates the mechanism of an unknown
    decoded formula, which would be circular.
    """
    # This file is written locally by prepare_mechanism_conditioned_data.py,
    # and contains tensors rather than a model checkpoint.
    prior = torch.load(path, weights_only=False, map_location="cpu")
    required = {"conditions", "condition_std", "mechanism_emb_zscore"}
    missing = required.difference(prior)
    if missing:
        raise RuntimeError(
            f"mechanism prior {path} misses required keys: {sorted(missing)}")
    conditions = torch.as_tensor(prior["conditions"], dtype=torch.float32)
    mechanism = torch.as_tensor(prior["mechanism_emb_zscore"], dtype=torch.float32)
    condition_std = torch.as_tensor(prior["condition_std"], dtype=torch.float32)
    if (conditions.ndim != 2 or conditions.shape[1] != 2
            or mechanism.shape != (conditions.shape[0], 2)
            or condition_std.shape != (2,) or torch.any(condition_std <= 0)):
        raise RuntimeError(
            "invalid mechanism prior shapes; expected conditions [N,2], "
            "mechanism_emb_zscore [N,2], condition_std [2]")
    if not torch.isfinite(conditions).all() or not torch.isfinite(mechanism).all():
        raise RuntimeError("mechanism prior contains non-finite values")
    return {"conditions": conditions, "condition_std": condition_std,
            "mechanism_emb_zscore": mechanism}


def sample_empirical_mechanism(prior, target_condition, batch_size, neighbors):
    """Sample a local empirical mechanism prior around one requested condition."""
    if neighbors < 1:
        raise ValueError("mechanism-prior-neighbors must be positive")
    target = target_condition.detach().cpu().reshape(1, 2)
    normalized_delta = ((prior["conditions"] - target)
                        / prior["condition_std"].reshape(1, 2))
    distance_sq = (normalized_delta ** 2).sum(dim=1)
    count = min(int(neighbors), distance_sq.numel())
    neighbor_indices = torch.topk(distance_sq, k=count, largest=False).indices
    draw = torch.randint(count, (batch_size,))
    return prior["mechanism_emb_zscore"][neighbor_indices[draw]]


def load_released_bamboo_generators(config_path, diffusion_checkpoint,
                                    decoder_checkpoint, bamboo_root,
                                    solvent_embeddings=None,
                                    salt_embeddings=None):
    """Load G0's released Bamboo checkpoint with its matching extension.

    The public Bamboo generation YAML has the historical two-field diffusion
    section, whereas the distributed checkpoint contains the later, fixed
    SigmaScheduler buffers.  The two missing scheduler fields are
    non-learnable and are reconstructed from that checkpoint's 1000-step
    schedule; the learned weights and released decoder remain unchanged.
    ``bamboo_root`` is retained in the CLI as provenance for the release.
    """
    del bamboo_root
    config = yaml.safe_load(Path(config_path).read_text())
    diffusion_config = dict(config["diffusion_config"]["model_config"])
    beta_config = diffusion_config["beta_scheduler_config"]
    diffusion_config.setdefault("sigma_scheduler_config", {
        "timesteps": beta_config["timesteps"],
        "sigma_begin": .01,
        "sigma_end": 1.0,
    })
    diffusion_config.setdefault("diff_config", {"time_dim": None})
    diffusion = FormulaDiffusion(**diffusion_config)
    diffusion.load_ckpt(diffusion_checkpoint)
    diffusion.eval().to("cuda")

    decoder_config = dict(config["decoder_config"]["model_config"])
    decoder_config["mol_dict"] = dict(decoder_config["mol_dict"])
    if solvent_embeddings:
        decoder_config["mol_dict"]["solv_dict_path"] = solvent_embeddings
    if salt_embeddings:
        decoder_config["mol_dict"]["salt_dict_path"] = salt_embeddings
    decoder = FormulaDecoder(**decoder_config)
    decoder.load_ckpt(decoder_checkpoint)
    return diffusion, decoder.eval().to("cuda")


def normalize_bow(bow, n_solvents):
    bow = torch.where(bow > .02, bow, torch.zeros_like(bow))
    for row in bow:
        if row[:n_solvents].sum() <= 0:
            row[row[:n_solvents].argmax()] = 1.0
        if row[n_solvents:].sum() <= 0:
            row[n_solvents + row[n_solvents:].argmax()] = 1.0
    left = torch.nn.functional.normalize(bow[:, :n_solvents], p=1, dim=-1)
    right = torch.nn.functional.normalize(bow[:, n_solvents:], p=1, dim=-1)
    return torch.cat([left, right], dim=-1)


def bow_to_formula(bow, solvents, salts, temperature, concentration):
    n_solvents = len(solvents)
    solvent_index = torch.where(bow[:n_solvents] > 0)[0].tolist()
    salt_index = torch.where(bow[n_solvents:] > 0)[0].tolist()
    return FormulaData(
        [solvents[i]["name"] for i in solvent_index],
        [Molecule.from_smiles(solvents[i]["smiles"]).get_mapped_smiles()
         for i in solvent_index],
        [salts[i]["name"] for i in salt_index],
        [Molecule.from_smiles(salts[i]["smiles"]).get_mapped_smiles()
         for i in salt_index],
        [bow[i].item() for i in solvent_index],
        [bow[n_solvents + i].item() for i in salt_index],
        temperature=temperature, concentration=concentration)


def score(oracle, bows, solvents, salts, temperature_c, concentration,
          batch_size, robust, max_conductivity):
    def one_condition(temp, conc):
        temp_norm = (temp + 273.15) / 373.15
        data = [bow_to_formula(bow, solvents, salts, temp_norm, conc)
                for bow in bows]
        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_data)
        conductivities, explain_residual, anion_ratios = [], [], []
        max_log_conductivity = np.log(max_conductivity)
        with torch.no_grad():
            for graph in loader:
                pred = oracle(graph.to("cuda"))
                # The oracle was trained on a finite conductivity range.  A
                # generator sample outside that range can yield an extreme
                # log-space prediction; exponentiating it would turn a model
                # extrapolation into ``Infinity``.  Preserve it as NaN here
                # so it is reported and excluded from ranking downstream.
                log_conductivity = (pred["conductivity"].squeeze(-1) *
                                    8.3344 - 4.6002)
                valid = (torch.isfinite(log_conductivity)
                         & (log_conductivity <= max_log_conductivity))
                physical = torch.full_like(log_conductivity, float("nan"))
                physical[valid] = torch.exp(log_conductivity[valid])
                conductivities.extend(physical.cpu().tolist())
                anion_ratios.extend(pred["anion_ratio"].squeeze(-1).cpu().tolist())
                explanation = pred.get("property_fragment_explanations")
                if explanation is not None:
                    residual = explanation["tasks"]["conductivity_explain"]["residual"]
                    explain_residual.extend(residual.cpu().tolist())
                else:
                    explain_residual.extend([0.0] * graph.counts.shape[0])
        return (np.asarray(conductivities), np.asarray(explain_residual),
                np.asarray(anion_ratios))

    primary, primary_explanation, primary_anion = one_condition(
        temperature_c, concentration)
    result = {"conductivity_25c_ms_cm": primary.tolist(),
              "explanation_residual_25c": primary_explanation.tolist(),
              "anion_ratio_pred": primary_anion.tolist()}
    if robust:
        grid_cond, grid_exp = [], []
        for temp in (25., 35., 45., 55.):
            for conc in (.05, .10, .15):
                values, effects, _ = one_condition(temp, conc)
                grid_cond.append(values)
                grid_exp.append(effects)
        grid_cond, grid_exp = np.asarray(grid_cond), np.asarray(grid_exp)
        result["robust_min_ms_cm"] = grid_cond.min(axis=0).tolist()
        result["robust_mean_ms_cm"] = grid_cond.mean(axis=0).tolist()
        result["robust_std_ms_cm"] = grid_cond.std(axis=0).tolist()
        signs = np.sign(grid_exp)
        result["explanation_sign_stability"] = (np.abs(signs.mean(axis=0))).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-config", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--decoder-solvent-embeddings")
    parser.add_argument("--decoder-salt-embeddings")
    parser.add_argument(
        "--mechanism-prior",
        help="mechanism_prior.pt for a 4-D mechanism-conditioned diffusion model")
    parser.add_argument(
        "--mechanism-prior-neighbors", type=int, default=256,
        help="number of nearest synthetic condition labels used by the "
             "empirical p(I_sigma, I_r | sigma, r) sampler")
    parser.add_argument("--released-bamboo-root",
                        help="Bamboo repository root; enables the original "
                             "released G0 generator implementation.")
    parser.add_argument("--solvents", required=True)
    parser.add_argument("--salts", required=True)
    parser.add_argument("--oracle-config", required=True)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-conductivity", type=float, default=20.)
    parser.add_argument("--anion-ratio", type=float, default=.5)
    parser.add_argument("--anion-ratio-min", type=float, default=0.0,
                        help="inclusive predicted anion-ratio eligibility lower bound")
    parser.add_argument("--anion-ratio-max", type=float, default=1.0,
                        help="inclusive predicted anion-ratio eligibility upper bound")
    parser.add_argument("--temperature-c", type=float, default=25.)
    parser.add_argument("--concentration", type=float, default=.1)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-conductivity", type=float, default=100.,
                        help="Physical upper guard (mS/cm); larger oracle "
                             "outputs are marked out-of-domain, not ranked.")
    parser.add_argument("--selection-policy",
                        choices=("conductivity_max", "joint_target"),
                        default="conductivity_max",
                        help="conductivity_max preserves the historical ranking. "
                        "joint_target ranks by normalized error to both requested "
                        "conductivity and anion ratio.")
    parser.add_argument("--conductivity-tolerance", type=float, default=2.0,
                        help="mS/cm error corresponding to one unit of joint "
                        "selection loss.")
    parser.add_argument("--anion-ratio-tolerance", type=float, default=.10,
                        help="anion-ratio error corresponding to one unit of joint "
                        "selection loss.")
    parser.add_argument("--joint-anion-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--robust", action="store_true")
    args = parser.parse_args()
    if args.conductivity_tolerance <= 0 or args.anion_ratio_tolerance <= 0:
        parser.error("joint-selection tolerances must be positive")
    if not (0.0 <= args.anion_ratio_min < args.anion_ratio_max <= 1.0):
        parser.error("require 0 <= --anion-ratio-min < --anion-ratio-max <= 1")
    if not args.anion_ratio_min <= args.anion_ratio <= args.anion_ratio_max:
        parser.error("--anion-ratio condition must be inside the eligibility window")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    solvents = json.loads(Path(args.solvents).read_text())
    salts = json.loads(Path(args.salts).read_text())
    if args.released_bamboo_root:
        diffusion, decoder = load_released_bamboo_generators(
            args.diffusion_config, args.diffusion_checkpoint,
            args.decoder_checkpoint, args.released_bamboo_root,
            args.decoder_solvent_embeddings, args.decoder_salt_embeddings)
    else:
        diffusion = load_generator(args.diffusion_config, args.diffusion_checkpoint,
                                   FormulaDiffusion, "diffusion_model_config")
        decoder_config = yaml.safe_load(Path(args.decoder_config).read_text())
        decoder_model_config = decoder_config.get("model", {}).get(
            "decoder_model_config")
        if decoder_model_config is None:
            decoder_model_config = decoder_config["decoder_config"]["model_config"]
        else:
            decoder_model_config = dict(decoder_model_config)
        decoder_model_config["mol_dict"] = dict(decoder_model_config["mol_dict"])
        if args.decoder_solvent_embeddings:
            decoder_model_config["mol_dict"]["solv_dict_path"] = args.decoder_solvent_embeddings
        if args.decoder_salt_embeddings:
            decoder_model_config["mol_dict"]["salt_dict_path"] = args.decoder_salt_embeddings
        decoder = FormulaDecoder(**decoder_model_config)
        decoder.load_ckpt(args.decoder_checkpoint)
        decoder.eval().to("cuda")
    property_condition = torch.tensor(
        [(np.log(args.target_conductivity) + 4.6002) / 8.3344,
         args.anion_ratio],
        dtype=torch.float32,
        device="cuda",
    )
    if diffusion.prop_dim == 4:
        if not args.mechanism_prior:
            parser.error(
                "this diffusion checkpoint has prop_dim=4; provide "
                "--mechanism-prior generated from its training synthetic set")
        mechanism_prior = load_mechanism_prior(args.mechanism_prior)
    elif diffusion.prop_dim == 2:
        if args.mechanism_prior:
            parser.error(
                "--mechanism-prior was supplied to a 2-D diffusion checkpoint")
        mechanism_prior = None
    else:
        raise RuntimeError(f"unsupported diffusion prop_dim={diffusion.prop_dim}")
    embeddings = []
    sampled_mechanisms = []
    with torch.no_grad():
        for index in range(args.num_batches):
            torch.manual_seed(args.seed + index)
            if mechanism_prior is None:
                condition = property_condition.repeat(args.batch_size, 1)
            else:
                mechanism = sample_empirical_mechanism(
                    mechanism_prior, property_condition, args.batch_size,
                    args.mechanism_prior_neighbors).to("cuda")
                condition = torch.cat((
                    property_condition.repeat(args.batch_size, 1), mechanism), dim=1)
                sampled_mechanisms.extend(mechanism.detach().cpu().tolist())
            embeddings.append(diffusion.sample(condition)[0]["frm_emb"])
        bow = decoder.predict(torch.cat(embeddings, dim=0).reshape(-1, 384))["bow_vec"]
    bow = normalize_bow(bow.detach().cpu(), len(solvents))
    scores = score(load_oracle(args.oracle_config, args.oracle_checkpoint), bow,
                   solvents, salts, args.temperature_c, args.concentration,
                   args.batch_size, args.robust, args.max_conductivity)
    records = []
    for index, row in enumerate(bow):
        conductivity = scores["conductivity_25c_ms_cm"][index]
        trusted = bool(np.isfinite(conductivity))
        if args.robust:
            trusted = trusted and all(np.isfinite(scores[name][index]) for name in (
                "robust_min_ms_cm", "robust_mean_ms_cm", "robust_std_ms_cm"))
        anion_ratio_pred = scores["anion_ratio_pred"][index]
        finite_anion_ratio = bool(np.isfinite(anion_ratio_pred))
        within_anion_window = (finite_anion_ratio and args.anion_ratio_min
                               <= anion_ratio_pred <= args.anion_ratio_max)
        oracle_valid = bool(np.isfinite(conductivity))
        # A numerically valid conductivity alone is not sufficient for a
        # trusted design candidate: the jointly requested transport window
        # must hold for the same decoded formula.
        trusted = trusted and within_anion_window
        record = {"bow": row.tolist(), "temperature_c": args.temperature_c,
                  "salt_molar_ratio": args.concentration,
                  "target_conductivity_ms_cm": args.target_conductivity,
                  "anion_ratio_condition": args.anion_ratio,
                  "anion_ratio_pred": (float(anion_ratio_pred)
                                       if finite_anion_ratio else None),
                  "anion_ratio_eligibility_min": args.anion_ratio_min,
                  "anion_ratio_eligibility_max": args.anion_ratio_max,
                  "within_anion_ratio_eligibility": bool(within_anion_window),
                  "oracle_conductivity_pred_ms_cm": (float(conductivity)
                                                      if oracle_valid else None),
                  "conductivity_pred_ms_cm": float(conductivity) if trusted else None,
                  "prediction_status": "trusted" if trusted else "out_of_domain",
                  "explanation_residual_25c": scores["explanation_residual_25c"][index],
                  "selection_policy": ("robust_min_conductivity" if args.robust
                                       else args.selection_policy)}
        if sampled_mechanisms:
            record["sampled_mechanism_condition_zscore"] = [
                float(value) for value in sampled_mechanisms[index]]
        for name in ("robust_min_ms_cm", "robust_mean_ms_cm", "robust_std_ms_cm",
                     "explanation_sign_stability"):
            if name in scores:
                value = scores[name][index]
                record[name] = float(value) if np.isfinite(value) else None
        if not finite_anion_ratio:
            record["eligibility_reason"] = "nonfinite_anion_ratio"
        elif not within_anion_window:
            record["eligibility_reason"] = "anion_ratio_outside_window"
        elif not oracle_valid:
            record["eligibility_reason"] = "conductivity_out_of_domain"
        else:
            record["eligibility_reason"] = ""
        # All groups are evaluated with a fixed downstream selection budget.
        # G0--G2 can retain the historical maximum-conductivity ranking or
        # rank by distance to the complete requested condition.  G3 instead
        # prioritizes the worst point in the 25--55 C,
        # 0.05--0.15 concentration grid and uses explanation-sign stability
        # only as a deterministic tie breaker.  This keeps the raw samples
        # unchanged while making the G3 policy explicit and auditable.
        if not trusted:
            record["candidate_rank_score"] = None
        elif args.robust:
            record["candidate_rank_score"] = (
                record["robust_min_ms_cm"]
                + 1e-3 * record["explanation_sign_stability"])
        elif args.selection_policy == "joint_target":
            if not finite_anion_ratio:
                record["candidate_rank_score"] = None
            else:
                conductivity_error = abs(
                    record["conductivity_pred_ms_cm"] - args.target_conductivity)
                anion_error = abs(record["anion_ratio_pred"] - args.anion_ratio)
                record["conductivity_target_abs_error_ms_cm"] = conductivity_error
                record["anion_ratio_abs_error"] = anion_error
                # Larger is better for the existing downstream sorter.  The
                # tolerances state the trade-off in physically interpretable
                # units, rather than silently letting conductivity dominate.
                record["candidate_rank_score"] = -(
                    conductivity_error / args.conductivity_tolerance
                    + args.joint_anion_weight
                    * anion_error / args.anion_ratio_tolerance)
        else:
            record["candidate_rank_score"] = record["conductivity_pred_ms_cm"]
        records.append(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
