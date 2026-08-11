"""Combine an untouched Bamboo mono backbone with a trained FragmentBlock."""

import argparse
import os

import torch


FRAGMENT_PREFIX = "graph_embed_block.fragment_block."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bamboo-mono", required=True)
    parser.add_argument("--fragment-mono", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bamboo = torch.load(
        args.bamboo_mono, map_location="cpu", weights_only=False)
    fragment = torch.load(
        args.fragment_mono, map_location="cpu", weights_only=False)
    bamboo_state = bamboo["model_state_dict"]
    fragment_state = fragment["model_state_dict"]
    fragment_only = {
        key: value for key, value in fragment_state.items()
        if key.startswith(FRAGMENT_PREFIX)
    }
    if not fragment_only:
        raise RuntimeError("fragment checkpoint contains no FragmentBlock")
    overlap = set(bamboo_state) & set(fragment_only)
    if overlap:
        raise RuntimeError(f"unexpected checkpoint key overlap: {sorted(overlap)}")

    combined = dict(bamboo_state)
    combined.update(fragment_only)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"model_state_dict": combined}, args.output)
    print(
        f"combined {len(bamboo_state)} Bamboo tensors with "
        f"{len(fragment_only)} FragmentBlock tensors: {args.output}")


if __name__ == "__main__":
    main()
