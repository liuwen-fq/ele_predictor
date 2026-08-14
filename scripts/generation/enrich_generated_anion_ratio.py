#!/usr/bin/env python3
"""Add frozen-oracle anion-ratio predictions to existing generated formulas.

This preserves prior diffusion samples and conductivity records.  It only
restores the output field written by Bamboo's original generation script but
previously omitted by the shared G0--G3 scoring path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# This file is executed from ``scripts/generation`` while the package lives
# at the repository root.  Insert it before importing the shared scorer.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from run_conditioned_generation import load_oracle, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--solvents", required=True)
    parser.add_argument("--salts", required=True)
    parser.add_argument("--oracle-config", required=True)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.input)
    records = json.loads(output.read_text())
    if not records:
        return
    solvents = json.loads(Path(args.solvents).read_text())
    salts = json.loads(Path(args.salts).read_text())
    bows = torch.tensor([row["bow"] for row in records], dtype=torch.float32)
    first = records[0]
    results = score(load_oracle(args.oracle_config, args.oracle_checkpoint),
                    bows, solvents, salts, float(first["temperature_c"]),
                    float(first["salt_molar_ratio"]), args.batch_size,
                    robust=False, max_conductivity=100.)
    for row, anion in zip(records, results["anion_ratio_pred"]):
        row["anion_ratio_pred"] = float(anion)
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Added anion_ratio_pred to {len(records)} records: {output}")


if __name__ == "__main__":
    main()
