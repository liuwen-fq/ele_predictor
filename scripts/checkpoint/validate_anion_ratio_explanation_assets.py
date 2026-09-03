#!/usr/bin/env python3
"""Fail fast if explanation-pipeline checkpoints do not match the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from formula_design.predictor.molmix import MolMix
from formula_design.predictor.mono import Mono


ALLOWED_BASE_MISSING_PREFIXES = (
    "graph_embed_block.graph_embed_block.fragment_block.",
    "property_fragment_explanation_head.",
)


def load_state(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain model_state_dict")
    return checkpoint["model_state_dict"]


def incompatible_keys(expected: dict, actual: dict) -> tuple[list[str], list[str], list[str]]:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    shape_mismatches = sorted(
        key for key in set(expected) & set(actual)
        if expected[key].shape != actual[key].shape)
    return missing, unexpected, shape_mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trained-checkpoint",
                        help="optional fully trained explanation checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    for label, path in (("base formula", config["check_point"]),
                        ("fixed mono", config["pretrain"]["ckpt_path"])):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {path}")

    mono = Mono(**config["pretrain"]["model_config"])
    mono_incompatible = mono.load_ckpt(config["pretrain"]["ckpt_path"])
    if mono_incompatible.missing_keys or mono_incompatible.unexpected_keys:
        raise RuntimeError("fixed mono checkpoint is not an exact architecture match")

    model = MolMix(mono, **config["model"])
    expected = model.state_dict()
    base = load_state(config["check_point"])
    missing, unexpected, shape_mismatches = incompatible_keys(expected, base)
    invalid_missing = [key for key in missing
                       if not key.startswith(ALLOWED_BASE_MISSING_PREFIXES)]
    if invalid_missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "base formula checkpoint mismatch: "
            f"invalid_missing={invalid_missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}")

    print("Checkpoint preflight passed")
    print(f"  fixed mono: {config['pretrain']['ckpt_path']} (exact match)")
    print(f"  base formula: {config['check_point']} "
          f"(compatible; {len(missing)} expected new/sidecar parameters omitted)")

    if args.trained_checkpoint:
        trained = load_state(args.trained_checkpoint)
        missing, unexpected, shape_mismatches = incompatible_keys(expected, trained)
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "trained explanation checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={shape_mismatches}")
        print(f"  trained explanation: {args.trained_checkpoint} (exact match)")


if __name__ == "__main__":
    main()
