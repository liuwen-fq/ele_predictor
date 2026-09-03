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

import json

import torch

from formula_design.data import Data, FormulaData
from formula_design.predictor.aggr import (
    AttentionAggr,
    ConditionAwareFragmentContribution,
    ConditionAwareFragmentExplainer,
    ConditionAwarePropertyFragmentExplainer,
    FragmentFormulaAggr,
    VisAggr,
)
from formula_design.predictor.graph_block import Graph2DBlock
from formula_design.predictor.tdep import TMDepBlock
from formula_design.utils import ActivationFunction

from .mono import Mono


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


class MolMix(torch.nn.Module):

    def __init__(self,
                 pretrained_model,
                 aggr_block: dict,
                 readout_block: dict,
                 anion_block: dict,
                 train_mode: str = "finetune",
                 formula_fragment_block: dict = None,
                 fragment_contribution_block: dict = None,
                 fragment_explanation_block: dict = None,
                 property_fragment_explanation_block: dict = None):
        super().__init__()

        self.graph_embed_block = pretrained_model
        # Formula-level training never updates molecular representations. This
        # makes Bamboo an immutable base and leaves residual learning to the
        # explicit contribution head below.
        for parameter in self.graph_embed_block.parameters():
            parameter.requires_grad = False
        self.graph_embed_block.eval()
        if train_mode not in {"pretrain", "finetune"}:
            raise ValueError(f"unsupported train_mode {train_mode}")
        self.train_mode = train_mode

        self.aggr_block_solv = AttentionAggr(**aggr_block)
        self.aggr_block_salt = AttentionAggr(**aggr_block, num_heads=2)
        self.aggr_vis = VisAggr()

        # MD pretraining supervises NE and Mistry conductivity separately.
        def make_heads():
            return torch.nn.ModuleDict({
                "sigma":
                ReadoutBlock(**readout_block, act="softplus"),
                "n1":
                ReadoutBlock(**readout_block, act="softplus"),
                "n2":
                ReadoutBlock(**readout_block, act="softplus"),
                "T0":
                ReadoutBlock(**readout_block, act="sigmoid"),
                "A":
                ReadoutBlock(**readout_block, act="softplus"),
                "B":
                ReadoutBlock(**readout_block, act="softplus"),
            })

        self.readout_ne = make_heads()
        self.readout_pred = make_heads()

        self.readout_anion_ratio = ReadoutBlock(**anion_block, act="sigmoid")

        self.empirical_formula_block = TMDepBlock()

        fragment_config = ({} if formula_fragment_block is None else
                           formula_fragment_block.copy())
        fragment_enabled = fragment_config.pop("enabled", False)
        freeze_base_formula = fragment_config.pop("freeze_base_formula", False)
        train_readout_pred = fragment_config.pop("train_readout_pred", False)
        self.fragment_formula_aggr = (FragmentFormulaAggr(
            **fragment_config) if fragment_enabled else None)

        contribution_config = ({}
                               if fragment_contribution_block is None else
                               fragment_contribution_block.copy())
        contribution_enabled = contribution_config.pop("enabled", False)
        contribution_freeze_base = contribution_config.pop(
            "freeze_base_formula", False)
        contribution_train_readout = contribution_config.pop(
            "train_readout_pred", False)
        if fragment_enabled and contribution_enabled:
            raise ValueError(
                "formula_fragment_block and fragment_contribution_block "
                "cannot be enabled together; their residuals would be "
                "scientifically ambiguous")
        self.fragment_contribution_head = (
            ConditionAwareFragmentContribution(**contribution_config)
            if contribution_enabled else None)
        explanation_config = ({} if fragment_explanation_block is None else
                              fragment_explanation_block.copy())
        explanation_enabled = explanation_config.pop("enabled", False)
        explanation_freeze_base = explanation_config.pop(
            "freeze_base_formula", False)
        explanation_train_readout = explanation_config.pop(
            "train_readout_pred", False)
        if sum((fragment_enabled, contribution_enabled,
                explanation_enabled)) > 1:
            raise ValueError(
                "formula_fragment_block, fragment_contribution_block and "
                "fragment_explanation_block are mutually exclusive")
        self.fragment_explanation_head = (
            ConditionAwareFragmentExplainer(**explanation_config)
            if explanation_enabled else None)
        property_explanation_config = (
            {} if property_fragment_explanation_block is None else
            property_fragment_explanation_block.copy())
        property_explanation_enabled = property_explanation_config.pop(
            "enabled", False)
        property_explanation_freeze_base = property_explanation_config.pop(
            "freeze_base_formula", False)
        property_explanation_train_readout = property_explanation_config.pop(
            "train_readout_pred", False)
        freeze_base_formula = freeze_base_formula or contribution_freeze_base
        train_readout_pred = train_readout_pred or contribution_train_readout
        freeze_base_formula = freeze_base_formula or explanation_freeze_base
        train_readout_pred = train_readout_pred or explanation_train_readout
        if sum((fragment_enabled, contribution_enabled, explanation_enabled,
                property_explanation_enabled)) > 1:
            raise ValueError(
                "formula_fragment_block, fragment_contribution_block, "
                "fragment_explanation_block and "
                "property_fragment_explanation_block are mutually exclusive")
        self.property_fragment_explanation_head = (
            ConditionAwarePropertyFragmentExplainer(
                **property_explanation_config)
            if property_explanation_enabled else None)
        freeze_base_formula = (freeze_base_formula
                               or property_explanation_freeze_base)
        train_readout_pred = (train_readout_pred
                              or property_explanation_train_readout)

        if freeze_base_formula:
            base_modules = (
                self.aggr_block_solv,
                self.aggr_block_salt,
                self.readout_ne,
                self.readout_pred,
                self.readout_anion_ratio,
            )
            for module in base_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = False
            if train_readout_pred:
                for parameter in self.readout_pred.parameters():
                    parameter.requires_grad = True

    def _get_device(self):
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.graph_embed_block.eval()
        return self

    def reset_parameters(self):
        self.aggr_block_solv.reset_parameters()
        self.aggr_block_salt.reset_parameters()
        for readout in (self.readout_ne, self.readout_pred):
            for layer in readout.values():
                layer.reset_parameters()
        self.readout_anion_ratio.reset_parameters()
        if self.fragment_formula_aggr is not None:
            self.fragment_formula_aggr.reset_parameters()
        if self.fragment_contribution_head is not None:
            self.fragment_contribution_head.reset_parameters()
        if self.fragment_explanation_head is not None:
            self.fragment_explanation_head.reset_parameters()
        if self.property_fragment_explanation_head is not None:
            self.property_fragment_explanation_head.reset_parameters()

    def get_parameters(self, name=None):
        return self.parameters() if name is None else getattr(
            self, name).parameters()

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def forward(self, data: Data):

        with torch.no_grad():
            solv_node_h, solv_edge_h = self.graph_embed_block._compute_embedding(
                data.solv_graphs)
            solv_fragment_diagnostics = (
                self.graph_embed_block.get_fragment_diagnostics())
            base_solv_properties = self.graph_embed_block.predict(
                data.solv_graphs, data.temperature, formula_vis=True)
            solv_vis = base_solv_properties["vis"]
            # The empirical Bamboo path uses its historical reference
            # viscosity.  The explanation path instead evaluates property
            # values at each formula's actual temperature without changing
            # that deployed prediction.
            solv_properties = base_solv_properties
            if self.property_fragment_explanation_head is not None:
                molecule_temperature = data.temperature.repeat_interleave(
                    data.solv_graphs.counts_mol[:, 0].long())
                solv_properties = self.graph_embed_block.predict(
                    data.solv_graphs,
                    molecule_temperature,
                    formula_vis=False,
                )
            salt_node_h, salt_edge_h = self.graph_embed_block._compute_embedding(
                data.salt_graphs)
            salt_fragment_diagnostics = (
                self.graph_embed_block.get_fragment_diagnostics())

        solv_node_h_inv, solv_edge_h_inv = self.aggr_block_solv(
            data.solv_graphs, solv_node_h, solv_edge_h)
        salt_node_h_inv, salt_edge_h_inv = self.aggr_block_salt(
            data.salt_graphs, salt_node_h, salt_edge_h)

        h_inv = torch.cat([
            solv_node_h_inv, solv_edge_h_inv, salt_node_h_inv, salt_edge_h_inv
        ],
                          dim=-1)
        vis = self.aggr_vis(data.solv_graphs, solv_vis)
        vis = vis.to(h_inv)
        T = data.temperature
        c = data.concentration

        h_ne = h_inv
        h_pred = h_inv
        formula_fragment_gates = None
        if self.fragment_formula_aggr is not None:
            if solv_fragment_diagnostics is None:
                raise RuntimeError(
                    "formula fragment sidecar requires a fragment-aware "
                    "molecular encoder")
            if solv_fragment_diagnostics["mode"] != "sidecar":
                raise RuntimeError(
                    "formula fragment sidecar requires fragment mode=sidecar")
            h_ne, h_pred = self.fragment_formula_aggr(
                data.solv_graphs,
                solv_fragment_diagnostics["molecule_embedding"],
                h_inv,
            )
            formula_fragment_gates = {
                name: value.detach()
                for name, value in
                self.fragment_formula_aggr.gate_values().items()
            }

        fragment_contributions = None
        fragment_explanations = None
        property_fragment_explanations = None
        if (self.fragment_contribution_head is not None
                or self.fragment_explanation_head is not None
                or self.property_fragment_explanation_head is not None):
            if solv_fragment_diagnostics is None:
                raise RuntimeError("fragment head requires a fragment-aware "
                                   "molecular encoder")
            if solv_fragment_diagnostics["mode"] != "sidecar":
                raise RuntimeError(
                    "fragment head requires fragment "
                    "mode=sidecar so Bamboo remains an explicit base model")
            condition_h = torch.cat([
                salt_node_h_inv,
                salt_edge_h_inv,
                c.reshape(-1, 1).to(h_inv),
                T.reshape(-1, 1).to(h_inv),
            ], dim=-1)
            fragment_args = (
                data.solv_graphs,
                solv_fragment_diagnostics["fragment_embedding"].to(h_inv),
                solv_fragment_diagnostics["fragment_batch"],
                condition_h,
                solv_fragment_diagnostics["fragment_active"],
            )
            if self.fragment_contribution_head is not None:
                fragment_contributions = self.fragment_contribution_head(
                    *fragment_args)
            if self.fragment_explanation_head is not None:
                fragment_explanations = self.fragment_explanation_head(
                    *fragment_args)
            if self.property_fragment_explanation_head is not None:
                property_names = (
                    self.property_fragment_explanation_head.property_names)
                solvent_property_h = torch.cat([
                    solv_properties[name].reshape(-1, 1).to(h_inv)
                    for name in property_names
                ], dim=-1)
                property_fragment_explanations = (
                    self.property_fragment_explanation_head(
                        data.solv_graphs,
                        solv_fragment_diagnostics["fragment_embedding"].to(
                            h_inv),
                        solv_fragment_diagnostics["fragment_batch"],
                        condition_h,
                        solvent_property_h,
                        fragment_active=solv_fragment_diagnostics[
                            "fragment_active"],
                    ))
                fragment_explanations = property_fragment_explanations

        # Empirical block inputs
        def get_inputs(readout, representation):
            return (
                readout["sigma"](representation),
                readout["n1"](representation),
                readout["n2"](representation),
                readout["T0"](representation),
                readout["A"](representation),
                readout["B"](representation),
            )

        h_all = torch.cat([
            h_inv,
            c.unsqueeze(-1).expand(-1, 8),
            T.unsqueeze(-1).expand(-1, 8)
        ],
                          dim=-1)
        anion_ratio = self.readout_anion_ratio(h_all)

        result = {
            "h_inv": h_inv,
            "anion_ratio": anion_ratio,
            "fragment_diagnostics": {
                "solvents": solv_fragment_diagnostics,
                "salts": salt_fragment_diagnostics,
                "formula_gates": formula_fragment_gates,
                "contributions": fragment_contributions,
                "explanations": fragment_explanations,
                "property_explanations": property_fragment_explanations,
                "solvent_properties": {
                    name: value.detach()
                    for name, value in solv_properties.items()
                },
            },
            "fragment_contributions": fragment_contributions,
            "fragment_explanations": fragment_explanations,
            "property_fragment_explanations": property_fragment_explanations,
        }
        if self.train_mode == "pretrain":
            conductivity_ne = self.empirical_formula_block(
                data, *get_inputs(self.readout_ne, h_ne), vis)
            conductivity_mistry = self.empirical_formula_block(
                data, *get_inputs(self.readout_pred, h_pred), vis)
            result["conductivity_NE_base"] = conductivity_ne
            result["conductivity_mistry_base"] = conductivity_mistry
            if fragment_contributions is not None:
                tasks = fragment_contributions["tasks"]
                conductivity_ne = conductivity_ne + tasks[
                    "conductivity_NE"]["residual"].unsqueeze(-1)
                conductivity_mistry = conductivity_mistry + tasks[
                    "conductivity_mistry"]["residual"].unsqueeze(-1)
            result["conductivity_NE"] = conductivity_ne
            result["conductivity_mistry"] = conductivity_mistry
        else:
            conductivity = self.empirical_formula_block(
                data, *get_inputs(self.readout_pred, h_pred), vis)
            result["conductivity_base"] = conductivity
            if fragment_contributions is not None:
                conductivity = conductivity + fragment_contributions[
                    "tasks"]["conductivity_mistry"]["residual"].unsqueeze(-1)
            result["conductivity"] = conductivity
            if fragment_explanations is not None:
                explain_tasks = fragment_explanations["tasks"]
                if "conductivity_explain" in explain_tasks:
                    explain_task = explain_tasks["conductivity_explain"]
                    result["conductivity_explain"] = explain_task[
                        "explanation_prediction"].unsqueeze(-1)
                    result["conductivity_explain_reference"] = explain_task[
                        "condition_reference"].unsqueeze(-1)
                if "anion_ratio_explain" in explain_tasks:
                    anion_explain_task = explain_tasks["anion_ratio_explain"]
                    result["anion_ratio_explain"] = anion_explain_task[
                        "explanation_prediction"].unsqueeze(-1)
                    # The additive constituents live in logit space; exposing
                    # it makes the salt baseline/property/fragment split
                    # interpretable without letting their sum leave [0, 1].
                    result["anion_ratio_explain_logit"] = anion_explain_task[
                        "explanation_logit"].unsqueeze(-1)
        return result
