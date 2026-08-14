#!/usr/bin/env python3
"""Write matched diffusion/decoder train configs for one experiment group."""

import argparse
from pathlib import Path

import yaml


def common(work, dataset_config, batch_size, max_epoch, lr):
    return {
        "meta": {"work_folder": str(work), "random_seed": 123,
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
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    dataset_config = root / "generator_data" / "dataset_config.yaml"
    vocab = root / "candidate_pool"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    diffusion = common(root / "train_diffusion", dataset_config,
                       args.batch_size, args.max_epoch, 1e-3)
    diffusion["dataset"][0]["loss"] = [{"loss_type": "diff_MSE", "weight": 1.0}]
    diffusion["model"] = {"model_type": "diffusion", "diffusion_model_config": {
        "embedding_dim": 384,
        "unet1d_config": {"dim": 64, "prop_dim": 2, "dim_mults": [1, 2, 4, 8],
                          "channels": 1, "dropout": 0.0, "self_condition": False,
                          "learned_variance": False, "learned_sinusoidal_cond": False,
                          "random_fourier_features": False,
                          "learned_sinusoidal_dim": 16,
                          "sinusoidal_pos_emb_theta": 10000, "attn_dim_head": 16,
                          "attn_heads": 4, "condition_method": "concat"},
        "beta_scheduler_config": {"timesteps": 1000, "scheduler_mode": "cosine"},
        "sigma_scheduler_config": {"timesteps": 1000}, "diff_config": {"time_dim": 256},
    }}
    decoder = common(root / "train_decoder", dataset_config,
                     args.batch_size, args.max_epoch, 5e-4)
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
