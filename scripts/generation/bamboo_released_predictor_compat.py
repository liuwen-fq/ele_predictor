"""Compatibility predictor for the released Bamboo formula checkpoint.

The distributed formula checkpoint has a single ``readout_conductivity``
ModuleDict, while the checked-out Bamboo source later split that dictionary
into ``readout_ne`` and ``readout_pred``.  This class reconstructs the model
topology encoded by the released state dict without changing any weights.
"""

import torch


def released_formula_predictor_class():
    from bamboo_mixer.predictor.aggr import AttentionAggr, VisAggr
    from bamboo_mixer.predictor.readout import ReadoutBlock
    from bamboo_mixer.predictor.tdep import TMDepBlock

    class ReleasedFormulaPredictor(torch.nn.Module):
        def __init__(self, mono_model, aggr_block, readout_block, anion_block,
                     train_mode="finetune"):
            super().__init__()
            if train_mode != "finetune":
                raise ValueError("released checkpoint is used only in finetune mode")
            self.graph_embed_block = mono_model
            self.graph_embed_block.eval()
            self.graph_embed_block.train = lambda mode=True: self.graph_embed_block
            self.aggr_block_solv = AttentionAggr(**aggr_block)
            self.aggr_block_salt = AttentionAggr(**aggr_block, num_heads=2)
            self.aggr_vis = VisAggr()
            self.readout_conductivity = torch.nn.ModuleDict({
                "sigma": ReadoutBlock(**readout_block, act="softplus"),
                "n1": ReadoutBlock(**readout_block, act="softplus"),
                "n2": ReadoutBlock(**readout_block, act="softplus"),
                "T0": ReadoutBlock(**readout_block, act="sigmoid"),
                "A": ReadoutBlock(**readout_block, act="softplus"),
                "B": ReadoutBlock(**readout_block, act="softplus"),
            })
            self.readout_anion_ratio = ReadoutBlock(**anion_block, act="sigmoid")
            self.empirical_formula_block = TMDepBlock()

        def load_ckpt(self, ckpt_path, device=None):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)["model_state_dict"]
            incompatible = self.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"Released formula checkpoint mismatch: {incompatible}")
            if device is not None:
                self.to(device)

        def forward(self, data, get_mol_embs=False):
            with torch.no_grad():
                solv_node_h, solv_edge_h = self.graph_embed_block.compute_embedding(data.solv_graphs)
                solv_vis = self.graph_embed_block.predict(data.solv_graphs, data.temperature, formula_vis=True)["vis"]
                salt_node_h, salt_edge_h = self.graph_embed_block.compute_embedding(data.salt_graphs)
            solv_node_h_inv, solv_edge_h_inv = self.aggr_block_solv(data.solv_graphs, solv_node_h, solv_edge_h)
            salt_node_h_inv, salt_edge_h_inv = self.aggr_block_salt(data.salt_graphs, salt_node_h, salt_edge_h)
            h_inv = torch.cat([solv_node_h_inv, solv_edge_h_inv, salt_node_h_inv, salt_edge_h_inv], dim=-1)
            vis = self.aggr_vis(data.solv_graphs, solv_vis).to(h_inv)
            head = self.readout_conductivity
            conductivity = self.empirical_formula_block(
                data, head["sigma"](h_inv), head["n1"](h_inv), head["n2"](h_inv),
                head["T0"](h_inv), head["A"](h_inv), head["B"](h_inv), vis)
            h_all = torch.cat([h_inv, data.concentration.unsqueeze(-1).expand(-1, 8),
                               data.temperature.unsqueeze(-1).expand(-1, 8)], dim=-1)
            result = {"h_inv": h_inv, "conductivity": conductivity,
                      "anion_ratio": self.readout_anion_ratio(h_all)}
            if get_mol_embs:
                result.update({"solv_node_h": solv_node_h, "solv_edge_h": solv_edge_h,
                               "salt_node_h": salt_node_h, "salt_edge_h": salt_edge_h})
            return result

    return ReleasedFormulaPredictor
