# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json

import torch
from torch_geometric.nn import global_mean_pool

from formula_design.data import Data
from formula_design.predictor.graph_block import Graph2DBlock
from formula_design.predictor.tdep import (LinearBlock, VFTBlock,
                                           VFTNoShiftBlock)
from formula_design.utils import ActivationFunction


class ReadoutBlock(torch.nn.Module):

    def __init__(self, act=None, **config):
        super(ReadoutBlock, self).__init__()

        # Dynamically retrieve values from config
        self.act = ActivationFunction.get_activation(act)
        self.input_dim = config.get('input_dim')
        self.hidden_dims = config.get('hidden_dims', [])
        self.output_dim = config.get('output_dim')

        # Create the readout layer dynamically
        self.readout_block = self.create_readout_block()

    def create_readout_block(self):
        layers = []
        input_dim = self.input_dim

        # Create layers based on hidden dimensions
        for hidden_dim in self.hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            input_dim = hidden_dim  # Update input dim for the next layer

        # Add the final output layer
        layers.append(torch.nn.Linear(input_dim, self.output_dim))

        if self.act:
            layers.append(self.act)

        # Return the layers as a Sequential block
        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.readout_block:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.readout_block(x)


class Mono(torch.nn.Module):

    def __init__(self, graph_block: dict, readout_block: dict):
        """

        readout_config example:
        config = {
            'input_dim': 64,
            'hidden_dims': [128, 64],
            'output_dim': 10
        }
        """
        super(Mono, self).__init__()
        self.graph_embed_block = Graph2DBlock(**graph_block)
        temp_readout_block = copy.deepcopy(readout_block)
        temp_readout_block['input_dim'] += 8  # cat temperature
        self.readout_Tm = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_bp = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_pka_a = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_pka_b = ReadoutBlock(**readout_block, act="sigmoid")
        self.readout_nD = ReadoutBlock(**temp_readout_block, act="sigmoid")
        self.readout_nD_liquid = ReadoutBlock(**temp_readout_block,
                                              act="sigmoid")
        self.readout_density = ReadoutBlock(**temp_readout_block,
                                            act="sigmoid")
        self.readout_dc = ReadoutBlock(**temp_readout_block, act="sigmoid")

        self.readout_vis_vis0 = ReadoutBlock(**readout_block, act="softplus")
        self.readout_vis_A = ReadoutBlock(**readout_block, act="softplus")
        self.readout_vis_T0 = ReadoutBlock(**readout_block, act="softplus")
        self.temp_vis = VFTBlock()

        self.readout_ST_A = ReadoutBlock(**readout_block, act="softplus")
        self.readout_ST_B = ReadoutBlock(**readout_block, act="softplus")
        self.temp_ST = LinearBlock()

        self.readout_vapP_A = ReadoutBlock(**readout_block, act="softplus")
        self.readout_vapP_B = ReadoutBlock(**readout_block, act="softplus")
        self.temp_vapP = VFTNoShiftBlock()

        if self.graph_embed_block.freeze_base:
            fragment_prefix = "graph_embed_block.fragment_block."
            for name, parameter in self.named_parameters():
                if not name.startswith(fragment_prefix):
                    parameter.requires_grad = False

    def reset_parameters(self):
        self.graph_embed_block.reset_parameters()
        self.readout_Tm.reset_parameters()
        self.readout_bp.reset_parameters()
        self.readout_nD.reset_parameters()
        self.readout_nD_liquid.reset_parameters()
        self.readout_pka_a.reset_parameters()
        self.readout_pka_b.reset_parameters()
        self.readout_density.reset_parameters()
        self.readout_dc.reset_parameters()
        self.readout_vis_vis0.reset_parameters()
        self.readout_vis_A.reset_parameters()
        self.readout_vis_T0.reset_parameters()
        self.readout_ST_A.reset_parameters()
        self.readout_ST_B.reset_parameters()
        self.readout_vapP_A.reset_parameters()
        self.readout_vapP_B.reset_parameters()

    def get_parameters(self, name=None):
        if name is None:
            return self.parameters()
        elif name == 'Graph':
            return self.graph_embed_block.parameters()

    def load_ckpt(self, ckpt_path):
        device = next(self.parameters()).device
        state_dict = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False,
        )['model_state_dict']
        strict = not self.graph_embed_block.fragment_enabled
        incompatible = self.load_state_dict(state_dict, strict=strict)
        if not strict:
            allowed_prefix = "graph_embed_block.fragment_block."
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith(allowed_prefix)
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "checkpoint is incompatible beyond the new fragment "
                    f"branch; missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}")
        return incompatible

    def _compute_embedding(self, data: Data):
        """
        """

        # 1. graph embedding

        node_h, edge_h = self.graph_embed_block(
            data)  # graph information and weigth ratio information

        return node_h, edge_h

    def get_fragment_diagnostics(self):
        """Return detached fragment-pooling diagnostics from the last forward."""

        fragment_block = self.graph_embed_block.fragment_block
        if fragment_block is None:
            return None
        if fragment_block.last_fragment_attention is None:
            raise RuntimeError(
                "fragment diagnostics requested before encoding")
        fragment_embedding, fragment_batch, fragment_active = (
            fragment_block.get_fragment_embeddings(detach=True))
        return {
            "attention":
            fragment_block.last_fragment_attention,
            "neutral_only":
            fragment_block.neutral_only,
            "mode":
            fragment_block.mode,
            "gate": (fragment_block.sidecar_gate().detach()
                     if fragment_block.mode == "sidecar" else None),
            "molecule_embedding":
            fragment_block.get_molecule_embedding(detach=True),
            "fragment_embedding":
            fragment_embedding,
            "fragment_batch":
            fragment_batch,
            "fragment_active":
            fragment_active,
        }

    def forward(self, data: Data):
        """
        """

        # 1. graph embedding

        node_h, edge_h = self.graph_embed_block(
            data)  # graph information and weigth ratio information

        device = node_h.device

        node_count, edge_count = data.counts[:, 0], data.counts[:, 1]

        node_batch_mol = torch.arange(
            node_count.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(node_count)

        edge_batch_mol = torch.arange(
            edge_count.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(edge_count)

        node_h_mean = global_mean_pool(node_h, node_batch_mol)
        edge_h_mean = global_mean_pool(edge_h, edge_batch_mol)

        h_all = torch.cat([node_h_mean, edge_h_mean], dim=-1)
        h_all = self.graph_embed_block.fuse_molecule_embedding(h_all)

        # Properties without temperature input
        Tm = self.readout_Tm(h_all)
        bp = self.readout_bp(h_all)
        pka_a = self.readout_pka_a(h_all)
        pka_b = self.readout_pka_b(h_all)

        # Properties with temperature

        h_nD = torch.cat([h_all, data.nD_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        nD = self.readout_nD(h_nD)

        h_nD_liquid = torch.cat(
            [h_all, data.nD_liquid_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        nD_liquid = self.readout_nD_liquid(h_nD_liquid)

        h_density = torch.cat(
            [h_all, data.density_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        density = self.readout_density(h_density)

        h_dc = torch.cat([h_all, data.dc_T.unsqueeze(-1).repeat(1, 8)], dim=-1)
        dc = self.readout_dc(h_dc)

        # Properties with temperature relation
        # viscosity (VFT)
        vis_vis0 = self.readout_vis_vis0(h_all)
        vis_A = self.readout_vis_A(h_all)
        vis_T0 = self.readout_vis_T0(h_all)
        vis_T = self.temp_vis(vis_vis0, vis_A, data.vis_T.unsqueeze(-1),
                              vis_T0)

        # ST (linear)
        ST_A = self.readout_ST_A(h_all)
        ST_B = self.readout_ST_B(h_all)
        ST_T = self.temp_ST(ST_A, ST_B, data.ST_T.unsqueeze(-1))

        # vapour pressure (VFTnonshift)
        vapP_A = self.readout_vapP_A(h_all)
        vapP_B = self.readout_vapP_B(h_all)
        vapP_T = self.temp_vapP(vapP_A, vapP_B, data.vapP_T.unsqueeze(-1))

        return {
            "Tm": Tm,
            "bp": bp,
            "nD": nD,
            "nD_liquid": nD_liquid,
            "pka_a": pka_a,
            "pka_b": pka_b,
            "density": density,
            "dc": dc,
            "vis": vis_T,
            "ST": ST_T,
            "vapP": vapP_T
        }

    def predict(self, data: Data, temperature, formula_vis=False):
        """
        """
        # 1. graph embedding

        node_h, edge_h = self.graph_embed_block(
            data)  # graph information and weigth ratio information

        device = node_h.device

        node_count, edge_count = data.counts[:, 0], data.counts[:, 1]

        node_batch_mol = torch.arange(
            node_count.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(node_count)

        edge_batch_mol = torch.arange(
            edge_count.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(edge_count)

        if formula_vis:
            node_count_formula = data.counts_mol[:, 0]
            temperature = temperature.repeat_interleave(node_count_formula,
                                                        dim=0)
            vis_temperature = 0.8 * torch.ones_like(temperature).unsqueeze(-1)
        else:
            vis_temperature = temperature.unsqueeze(-1) * 373.15 / 404.1

        node_h_mean = global_mean_pool(node_h, node_batch_mol)
        edge_h_mean = global_mean_pool(edge_h, edge_batch_mol)

        h_all = torch.cat([node_h_mean, edge_h_mean], dim=-1)

        # Properties without temperature input
        Tm = self.readout_Tm(h_all)
        bp = self.readout_bp(h_all)
        pka_a = self.readout_pka_a(h_all)
        pka_b = self.readout_pka_b(h_all)

        nD_T = temperature.unsqueeze(-1) * 373.15 / 407.15
        h_nD = torch.cat([h_all, nD_T.repeat(1, 8)], dim=-1)
        nD = self.readout_nD(h_nD)

        nD_liquid_T = temperature.unsqueeze(-1) * 373.15 / 407.15
        h_nD_liquid = torch.cat([h_all, nD_liquid_T.repeat(1, 8)], dim=-1)
        nD_liquid = self.readout_nD_liquid(h_nD_liquid)

        density_T = temperature.unsqueeze(-1) * 373.15 / 548.15
        h_density = torch.cat([h_all, density_T.repeat(1, 8)], dim=-1)
        density = self.readout_density(h_density)

        dc_T = temperature.unsqueeze(-1) * 373.15 / 502.15
        h_dc = torch.cat([h_all, dc_T.repeat(1, 8)], dim=-1)
        dc = self.readout_dc(h_dc)

        # Properties with temperature relation
        # viscosity (VFT)
        vis_vis0 = self.readout_vis_vis0(h_all)
        vis_A = self.readout_vis_A(h_all)
        vis_T0 = self.readout_vis_T0(h_all)
        vis_T = self.temp_vis(vis_vis0, vis_A, vis_temperature, vis_T0)

        # ST (linear)
        ST_A = self.readout_ST_A(h_all)
        ST_B = self.readout_ST_B(h_all)
        ST_temperature = temperature.unsqueeze(
            -1) * 373.15 / 473.15  # convert to the scale for pretrained model
        ST_T = self.temp_ST(ST_A, ST_B, ST_temperature)

        # vapour pressure (VFTnonshift)
        vapP_A = self.readout_vapP_A(h_all)
        vapP_B = self.readout_vapP_B(h_all)
        vapP_temperature = temperature.unsqueeze(
            -1) * 373.15 / 473.15  # convert to the scale for pretrained model
        vapP_T = self.temp_vapP(vapP_A, vapP_B, vapP_temperature)

        return {
            "Tm": Tm,
            "bp": bp,
            "nD": nD,
            "nD_liquid": nD_liquid,
            "pka_a": pka_a,
            "pka_b": pka_b,
            "density": density,
            "dc": dc,
            "vis": vis_T,
            "ST": ST_T,
            "vapP": vapP_T
        }
