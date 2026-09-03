#!/usr/bin/env python3
"""Write matched diffusion/decoder train configs for one experiment group."""

import argparse
from pathlib import Path

import yaml


MECHANISM_AUXILIARY_SCHEMAS = {
    "sigma": ["I_sigma_conditional_residual_zscore"],
    "r": ["I_r_conditional_residual_zscore"],
    "sigma_r": ["I_sigma_conditional_residual_zscore",
                "I_r_conditional_residual_zscore"],
}


def common(work, dataset_config, batch_size, max_epoch, lr, seed):
    return {
        "meta": {"work_folder": str(work), "random_seed": seed,
                 "fp64": False, "print_grad": False,
                 "save_debug_data": False},
        "dataset": [{"config": str(dataset_config), "batch_size": batch_size,
                     "shuffle": True, "train_ratio": 0.7, "loss_weight": 1.0,
                     "num_workers": 4, "prefetch_factor": 2}],
        "training": {"valid_interval": 2, "ckpt_interval": 50,
                     "max_epoch": max_epoch, "amp": False, "grad_clip": 5.0,
                     "optimizer": {"type": "RAdam", "lr": lr},
                     "scheduler": {"type": "ReduceLROnPlateau", "factor": .9,
                                   "patience": 3, "threshold_mode": "abs",
                                   "threshold": 1e-6, "min_lr": 1e-7},
                     "early_stop_patience": 20, "ignore_tolerance": 1e-6},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epoch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed recorded in both trainer configs.")
    parser.add_argument(
        "--mechanism-conditioning", action="store_true",
        help="Use the explicit 4-D diffusion condition "
            "[sigma, anion_ratio, I_sigma, I_r].  The decoder remains "
            "property-agnostic.")
    parser.add_argument(
        "--mechanism-auxiliary", action="store_true",
        help="Keep the 2-D [sigma, anion_ratio] generation condition and "
        "train a denoising auxiliary head against formula-level interaction "
        "residual labels in data.mechanism_aux_emb.")
    parser.add_argument(
        "--mechanism-auxiliary-weight", type=float, default=.05,
        help="Weight of the SNR-weighted Huber mechanism auxiliary loss.")
    parser.add_argument(
        "--mechanism-auxiliary-hidden-dim", type=int, default=192,
        help="Hidden width of the mechanism auxiliary prediction head.")
    parser.add_argument(
        "--mechanism-auxiliary-targets",
        choices=tuple(MECHANISM_AUXILIARY_SCHEMAS), default="sigma_r",
        help="Auxiliary residual labels: sigma, r, or sigma_r. Must match "
             "the prepared generator_data/mechanism_aux_emb dimension.")
    args = parser.parse_args()
    if args.mechanism_conditioning and args.mechanism_auxiliary:
        parser.error(
            "--mechanism-conditioning and --mechanism-auxiliary are "
            "separate experiments and cannot be enabled together")
    if args.mechanism_auxiliary_weight < 0:
        parser.error("--mechanism-auxiliary-weight must be non-negative")
    if args.mechanism_auxiliary_hidden_dim < 1:
        parser.error("--mechanism-auxiliary-hidden-dim must be positive")
    mechanism_auxiliary_schema = MECHANISM_AUXILIARY_SCHEMAS[
        args.mechanism_auxiliary_targets]
    root = Path(args.experiment_dir)
    dataset_config = root / "generator_data" / "dataset_config.yaml"
    vocab = root / "candidate_pool"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    diffusion = common(root / "train_diffusion", dataset_config,
                       args.batch_size, args.max_epoch, 1e-3, args.seed)
    diffusion["dataset"][0]["loss"] = [{"loss_type": "diff_MSE", "weight": 1.0}]
    if args.mechanism_auxiliary:
        diffusion["dataset"][0]["loss"].append({
            "loss_type": "mechanism_aux_Huber",
            "weight": args.mechanism_auxiliary_weight,
            "kwargs": {"snr_weighted": True},
        })
    prop_dim = 4 if args.mechanism_conditioning else 2
    diffusion["model"] = {"model_type": "diffusion", "diffusion_model_config": {
        "embedding_dim": 384,
        "unet1d_config": {"dim": 64, "prop_dim": prop_dim, "dim_mults": [1, 2, 4, 8],
                          "channels": 1, "dropout": 0.0, "self_condition": False,
                          "learned_variance": False, "learned_sinusoidal_cond": False,
                          "random_fourier_features": False,
                          "learned_sinusoidal_dim": 16,
                          "sinusoidal_pos_emb_theta": 10000, "attn_dim_head": 16,
                          "attn_heads": 4, "condition_method": "concat"},
        "beta_scheduler_config": {"timesteps": 1000, "scheduler_mode": "cosine"},
        "sigma_scheduler_config": {"timesteps": 1000},
        "diff_config": {
            "time_dim": 256,
            "mechanism_auxiliary": {
                "enabled": bool(args.mechanism_auxiliary),
                "label_dim": len(mechanism_auxiliary_schema),
                "hidden_dim": args.mechanism_auxiliary_hidden_dim,
            },
        },
    }}
    diffusion["mechanism_conditioning"] = {
        "enabled": bool(args.mechanism_conditioning),
        "condition_schema": (
            ["conductivity_normalized", "anion_ratio", "I_sigma_zscore", "I_r_zscore"]
            if args.mechanism_conditioning else
            ["conductivity_normalized", "anion_ratio"]),
    }
    diffusion["mechanism_auxiliary"] = {
        "enabled": bool(args.mechanism_auxiliary),
        "label_field": "mechanism_aux_emb" if args.mechanism_auxiliary else None,
        "feature_schema": (
            mechanism_auxiliary_schema
            if args.mechanism_auxiliary else None),
        "loss": ("snr_weighted_huber" if args.mechanism_auxiliary else None),
        "loss_weight": (
            args.mechanism_auxiliary_weight if args.mechanism_auxiliary else None),
    }
    decoder = common(root / "train_decoder", dataset_config,
                     args.batch_size, args.max_epoch, 5e-4, args.seed)
    decoder["dataset"][0]["loss"] = [{"loss_type": "bow_BCE", "weight": 1.0},
                                        {"loss_type": "bow_MAE", "weight": 0.0}]
    decoder["model"] = {"model_type": "decoder", "decoder_model_config": {
        "formula_emb_dim": 384,
        "set_transformer_block": {"hidden_dim": 64, "max_num_mol": 12,
                                  "num_layers": 2},
        "aggr_block": {"node_emb_dim": 32, "node_att_dim": 32,
                       "edge_emb_dim": 32, "edge_att_dim": 32},
        "mol_dict": {"solv_dict_path": str(vocab / "solv_embs.pt"),
                     "salt_dict_path": str(vocab / "salt_embs.pt")},
    }}
    (config_dir / "diffusion.yaml").write_text(yaml.safe_dump(diffusion, sort_keys=False))
    (config_dir / "decoder.yaml").write_text(yaml.safe_dump(decoder, sort_keys=False))


if __name__ == "__main__":
    main()
