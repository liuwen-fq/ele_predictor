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
from formula_design.predictor import MolMix, Mono


def load_oracle(config_path, checkpoint):
    config = yaml.safe_load(Path(config_path).read_text())
    mono = Mono(**config["model"]["pretrain"]["model_config"])
    mono.load_ckpt(str(Path(checkpoint) / "pretrain.pt"))
    model = MolMix(mono, **config["model"]["model"])
    model.load_ckpt(str(Path(checkpoint) / "optimal.pt"))
    return model.eval().to("cuda")


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
    parser.add_argument("--temperature-c", type=float, default=25.)
    parser.add_argument("--concentration", type=float, default=.1)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-conductivity", type=float, default=100.,
                        help="Physical upper guard (mS/cm); larger oracle "
                             "outputs are marked out-of-domain, not ranked.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--robust", action="store_true")
    args = parser.parse_args()
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
    condition = torch.tensor(
        [(np.log(args.target_conductivity) + 4.6002) / 8.3344,
         args.anion_ratio],
        dtype=torch.float32,
        device="cuda",
    ).repeat(args.batch_size, 1)
    embeddings = []
    with torch.no_grad():
        for index in range(args.num_batches):
            torch.manual_seed(args.seed + index)
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
        record = {"bow": row.tolist(), "temperature_c": args.temperature_c,
                  "salt_molar_ratio": args.concentration,
                  "target_conductivity_ms_cm": args.target_conductivity,
                  "anion_ratio_condition": args.anion_ratio,
                  "anion_ratio_pred": float(scores["anion_ratio_pred"][index]),
                  "conductivity_pred_ms_cm": float(conductivity) if trusted else None,
                  "prediction_status": "trusted" if trusted else "out_of_domain",
                  "explanation_residual_25c": scores["explanation_residual_25c"][index]}
        for name in ("robust_min_ms_cm", "robust_mean_ms_cm", "robust_std_ms_cm",
                     "explanation_sign_stability"):
            if name in scores:
                value = scores[name][index]
                record[name] = float(value) if np.isfinite(value) else None
        # All groups are evaluated with a fixed downstream selection budget.
        # G0--G2 rank by the frozen-oracle conductivity at the requested
        # condition.  G3 instead prioritizes the worst point in the 25--55 C,
        # 0.05--0.15 concentration grid and uses explanation-sign stability
        # only as a deterministic tie breaker.  This keeps the raw samples
        # unchanged while making the G3 policy explicit and auditable.
        if not trusted:
            record["candidate_rank_score"] = None
        elif args.robust:
            record["candidate_rank_score"] = (
                record["robust_min_ms_cm"]
                + 1e-3 * record["explanation_sign_stability"])
        else:
            record["candidate_rank_score"] = record["conductivity_pred_ms_cm"]
        records.append(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
