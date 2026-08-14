#!/usr/bin/env python3
"""Embed one candidate vocabulary with the frozen molecular encoder."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.nn import global_mean_pool

from formula_design.data import MonoData, collate_data
from formula_design.mol import Molecule
from formula_design.predictor import Mono


def load_model(config_path, checkpoint):
    config = yaml.safe_load(Path(config_path).read_text())
    model = Mono(**config["model"]["pretrain"]["model_config"])
    model.load_ckpt(checkpoint)
    return model.eval().to("cuda")


def embed(entries, model, batch_size):
    output = []
    for start in range(0, len(entries), batch_size):
        entries_part = entries[start:start + batch_size]
        data = [MonoData(item["name"], Molecule.from_smiles(
            item["smiles"]).get_mapped_smiles()) for item in entries_part]
        with torch.no_grad():
            graph = collate_data(data).to("cuda")
            node_h, edge_h = model._compute_embedding(graph)
            node_batch = torch.arange(graph.counts.shape[0], device="cuda").repeat_interleave(
                graph.counts[:, 0].long())
            edge_batch = torch.arange(graph.counts.shape[0], device="cuda").repeat_interleave(
                graph.counts[:, 1].long())
            h = torch.cat([global_mean_pool(node_h, node_batch),
                           global_mean_pool(edge_h, edge_batch)], dim=-1)
        output.append(h.cpu())
    return torch.cat(output, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvents", required=True)
    parser.add_argument("--salts", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--mono-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    solvents = json.loads(Path(args.solvents).read_text())
    salts = json.loads(Path(args.salts).read_text())
    model = load_model(args.model_config, args.mono_checkpoint)
    torch.save(embed(solvents, model, args.batch_size), output / "solv_embs.pt")
    torch.save(embed(salts, model, args.batch_size), output / "salt_embs.pt")


if __name__ == "__main__":
    main()
