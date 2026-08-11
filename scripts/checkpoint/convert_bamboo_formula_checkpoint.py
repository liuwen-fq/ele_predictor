"""Convert Bamboo-Mixer formula-head names to ele_predictor names."""

import argparse
import os

import torch

HEAD_NAME_MAP = {
    "Asigma0": "sigma",
    "B": "A",
    "D": "B",
}


def convert_key(key: str) -> str:
    parts = key.split(".")
    if parts[0] in {"readout_ne", "readout_pred"} and len(parts) > 1:
        parts[1] = HEAD_NAME_MAP.get(parts[1], parts[1])
    return ".".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    converted = {convert_key(key): value for key, value in state_dict.items()}
    if len(converted) != len(state_dict):
        raise RuntimeError("checkpoint key conversion produced duplicates")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"model_state_dict": converted}, args.output)
    print(f"converted {len(converted)} tensors: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
