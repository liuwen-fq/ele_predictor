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
import math

import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

from formula_design.utils import to_dense_batch


class AttentionBlock(torch.nn.Module):

    def __init__(self, emb_dim, att_dim):
        super().__init__()

        self.att_q_net = torch.nn.Linear(emb_dim, att_dim)
        self.att_k_net = torch.nn.Linear(emb_dim, att_dim)
        self.att_v_net = torch.nn.Linear(emb_dim, emb_dim)

    def reset_parameters(self):
        self.att_q_net.reset_parameters()
        self.att_k_net.reset_parameters()
        self.att_k_net.reset_parameters()

    def forward(self, x_h):
        q = self.att_q_net(x_h)  #(batch_size, n_mols,  att_dim)
        k = self.att_k_net(x_h)  # (batch_size, n_mols,  att_dim)
        v = self.att_v_net(x_h)  # (batch_size, n_mols,  emb_dim)

        att_scores = torch.bmm(q, k.transpose(
            1, 2)) / k.size(2)**0.5  # (batch_size, n_mols, n_mols)
        att_outputs = torch.bmm(torch.softmax(att_scores, dim=1),
                                v)  # (batch_size, n_mols,  emb_dim)

        return att_outputs


class MultiHeadAttentionBlock(torch.nn.Module):

    def __init__(self, emb_dim, att_dim, num_heads):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionBlock(emb_dim, att_dim) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class AttentionAggr(torch.nn.Module):

    def __init__(self,
                 node_emb_dim=32,
                 node_att_dim=32,
                 edge_emb_dim=32,
                 edge_att_dim=32,
                 num_heads=4):
        super().__init__()

        self.node_attention_block = MultiHeadAttentionBlock(
            node_emb_dim, node_att_dim, num_heads)
        self.edge_attention_block = MultiHeadAttentionBlock(
            edge_emb_dim, edge_att_dim, num_heads)
        self.node_norm = torch.nn.LayerNorm(node_att_dim * num_heads)
        self.edge_norm = torch.nn.LayerNorm(edge_att_dim * num_heads)

    def reset_parameters(self):
        self.node_attention_block.reset_parameters()
        self.edge_attention_block.reset_parameters()

    def forward(self, data, node_h, edge_h):

        device = node_h.device

        # Pooling the embedding for each molecule in mixture

        node_count, edge_count = data.counts[:, 0], data.counts[:, 1]

        node_batch_mol = torch.arange(
            node_count.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(node_count)

        edge_batch_mol = torch.arange(
            edge_count.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(edge_count)

        node_h_mol = global_mean_pool(node_h, node_batch_mol)
        edge_h_mol = global_mean_pool(edge_h, edge_batch_mol)

        # unsqueeze the batch size dimension to account for "interaction" between different component molecules in the electrolyte system
        node_count_formula, edge_count_formula = data.counts_mol[:,
                                                                 0], data.counts_mol[:,
                                                                                     1]
        node_batch_formula = torch.arange(
            node_count_formula.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(node_count_formula)
        edge_batch_formula = torch.arange(
            edge_count_formula.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(edge_count_formula)

        node_h_formula, _ = to_dense_batch(node_h_mol,
                                           node_batch_formula,
                                           fill_value=0.0,
                                           max_num_nodes=12)
        edge_h_formula, _ = to_dense_batch(edge_h_mol,
                                           edge_batch_formula,
                                           fill_value=0.0,
                                           max_num_nodes=12)

        # # unsqueeze the batch size dimension for weight ratios

        molar_ratios, _ = to_dense_batch(data.molar_ratios,
                                         node_batch_formula,
                                         fill_value=0.0,
                                         max_num_nodes=12)

        node_h_formula = molar_ratios * node_h_formula
        edge_h_formula = molar_ratios * edge_h_formula

        node_outputs = self.node_attention_block(node_h_formula)
        edge_outputs = self.edge_attention_block(edge_h_formula)

        # Remove only the pooled-molecule axis.  A bare squeeze would also
        # remove the formula batch axis for the final one-formula batch.
        node_embedding_inv = torch.bmm(molar_ratios.transpose(1, 2),
                                       node_outputs).squeeze(1)
        edge_embedding_inv = torch.bmm(molar_ratios.transpose(1, 2),
                                       edge_outputs).squeeze(1)

        node_embedding_inv = self.node_norm(node_embedding_inv)
        edge_embedding_inv = self.edge_norm(edge_embedding_inv)

        return node_embedding_inv, edge_embedding_inv


class VisAggr(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data, vis):

        device = vis.device

        # unsqueeze the batch size dimension to account for "interaction" between different component molecules in the electrolyte system
        node_count_formula = data.counts_mol[:, 0]
        node_batch_formula = torch.arange(
            node_count_formula.size()[0], dtype=torch.int64,
            device=device).repeat_interleave(node_count_formula)

        # # unsqueeze the batch size dimension for weight ratios

        molar_ratios, _ = to_dense_batch(data.molar_ratios,
                                         node_batch_formula,
                                         fill_value=0.0,
                                         max_num_nodes=12)

        vis_batch, _ = to_dense_batch(vis,
                                      node_batch_formula,
                                      fill_value=0.0,
                                      max_num_nodes=12)

        # ``TMDepBlock`` appends its own last dimension to ``vis``.  Return
        # one scalar per formula, i.e. ``(batch,)`` also when batch=1; keeping
        # ``(batch, 1)`` here would later broadcast conductivity to
        # ``(batch, batch, 1)``.
        vis_mix = torch.bmm(
            molar_ratios.transpose(1, 2), vis_batch).squeeze(-1).squeeze(-1)

        return vis_mix


class FragmentFormulaAggr(torch.nn.Module):
    """Aggregate solvent-level sidecar embeddings into formula corrections."""

    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 output_dim: int = 384,
                 gate_scale: float = 0.1,
                 gate_init: float = 0.0):
        super().__init__()
        if gate_scale <= 0:
            raise ValueError("gate_scale must be positive")
        self.gate_scale = float(gate_scale)
        self.gate_init = float(gate_init)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.ne_projection = torch.nn.Linear(hidden_dim, output_dim)
        self.pred_projection = torch.nn.Linear(hidden_dim, output_dim)
        self.ne_gate_logit = torch.nn.Parameter(torch.tensor(self.gate_init))
        self.pred_gate_logit = torch.nn.Parameter(torch.tensor(self.gate_init))

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        with torch.no_grad():
            self.ne_gate_logit.fill_(self.gate_init)
            self.pred_gate_logit.fill_(self.gate_init)

    def _pool_formula(self, data, molecule_h):
        molecule_counts = data.counts_mol[:, 0]
        formula_batch = torch.arange(
            molecule_counts.shape[0],
            dtype=torch.long,
            device=molecule_h.device,
        ).repeat_interleave(molecule_counts.long())
        ratios = data.molar_ratios.to(molecule_h).reshape(-1, 1)
        return scatter(
            molecule_h * ratios,
            formula_batch,
            dim=0,
            dim_size=molecule_counts.shape[0],
            reduce="sum",
        )

    def forward(self, data, molecule_h, base_h):
        formula_h = self.encoder(self._pool_formula(data, molecule_h))
        ne_gate = self.gate_scale * torch.tanh(self.ne_gate_logit)
        pred_gate = self.gate_scale * torch.tanh(self.pred_gate_logit)
        h_ne = base_h + ne_gate * self.ne_projection(formula_h)
        h_pred = base_h + pred_gate * self.pred_projection(formula_h)
        return h_ne, h_pred

    def gate_values(self):
        return {
            "conductivity_NE":
            self.gate_scale * torch.tanh(self.ne_gate_logit),
            "conductivity_mistry":
            self.gate_scale * torch.tanh(self.pred_gate_logit),
        }


class ConditionAwareFragmentContribution(torch.nn.Module):
    """Additive, signed fragment effects and fragment-pair interactions.

    The module predicts a residual in the normalized log-conductivity space.
    Bamboo's formulation prediction remains an explicit base term.  Every
    residual is decomposed exactly into molar-ratio-weighted fragment main
    effects and low-rank pair interactions conditioned on salt representation,
    concentration, and temperature.
    """

    DEFAULT_TASKS = ("conductivity_NE", "conductivity_mistry")

    def __init__(self,
                 input_dim: int = 64,
                 condition_input_dim: int = 130,
                 hidden_dim: int = 64,
                 main_rank: int = 16,
                 pair_rank: int = 8,
                 main_scale: float = 0.1,
                 interaction_scale: float = 0.05,
                 residual_bound: float = 0.06,
                 cross_molecule_only: bool = True,
                 diagnostic_top_k_pairs: int = 20,
                 task_names=None):
        super().__init__()
        if min(input_dim, condition_input_dim, hidden_dim, main_rank,
               pair_rank) <= 0:
            raise ValueError("all contribution-head dimensions must be positive")
        if main_scale <= 0 or interaction_scale <= 0 or residual_bound <= 0:
            raise ValueError("contribution scales must be positive")
        if diagnostic_top_k_pairs < -1:
            raise ValueError(
                "diagnostic_top_k_pairs must be -1 or non-negative")

        self.input_dim = int(input_dim)
        self.condition_input_dim = int(condition_input_dim)
        self.main_rank = int(main_rank)
        self.pair_rank = int(pair_rank)
        self.main_scale = float(main_scale)
        self.interaction_scale = float(interaction_scale)
        self.residual_bound = float(residual_bound)
        self.cross_molecule_only = bool(cross_molecule_only)
        self.diagnostic_top_k_pairs = int(diagnostic_top_k_pairs)
        self.task_names = tuple(task_names or self.DEFAULT_TASKS)
        if not self.task_names:
            raise ValueError("at least one contribution task is required")

        self.fragment_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(condition_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.main_fragment = torch.nn.Linear(hidden_dim, main_rank)
        self.pair_fragment = torch.nn.Linear(hidden_dim, pair_rank)
        self.main_condition = torch.nn.ModuleDict({
            task: torch.nn.Linear(hidden_dim, main_rank)
            for task in self.task_names
        })
        self.pair_condition = torch.nn.ModuleDict({
            task: torch.nn.Linear(hidden_dim, pair_rank)
            for task in self.task_names
        })
        self.reset_parameters()

    def reset_parameters(self):
        for module in (
                self.fragment_encoder,
                self.condition_encoder,
                self.main_fragment,
                self.pair_fragment,
        ):
            for child in module.modules() if isinstance(
                    module, torch.nn.Sequential) else (module, ):
                if child is not module and hasattr(child, "reset_parameters"):
                    child.reset_parameters()
                elif child is module and hasattr(child, "reset_parameters"):
                    child.reset_parameters()
        # Zero context projections make a newly attached head an exact Bamboo
        # identity while retaining first-step gradients for these projections.
        for projection in list(self.main_condition.values()) + list(
                self.pair_condition.values()):
            torch.nn.init.zeros_(projection.weight)
            torch.nn.init.zeros_(projection.bias)

    @staticmethod
    def _batch_index(counts, device):
        return torch.arange(
            counts.shape[0], dtype=torch.long,
            device=device).repeat_interleave(counts.long())

    def _assignments(self, data, fragment_batch, fragment_active,
                     dtype):
        molecule_counts = data.counts_mol[:, 0]
        molecule_formula = self._batch_index(molecule_counts,
                                             fragment_batch.device)
        if fragment_batch.numel() and int(fragment_batch.max()) >= len(
                molecule_formula):
            raise ValueError("fragment-to-molecule assignment is out of range")
        fragment_formula = molecule_formula[fragment_batch]
        molecule_ratio = data.molar_ratios.to(
            device=fragment_batch.device, dtype=dtype).reshape(-1)
        fragment_ratio = molecule_ratio[fragment_batch]
        if fragment_active is None:
            fragment_active = fragment_ratio.new_ones(fragment_ratio.shape)
        else:
            fragment_active = fragment_active.to(fragment_ratio)
        # Divide every fragment by the total number of fragments in its
        # molecule.  The denominator deliberately ignores fragment_active:
        # masking one fragment removes its signal instead of redistributing
        # that signal over the fragments that remain.
        molecule_fragment_count = scatter(
            torch.ones_like(fragment_ratio),
            fragment_batch,
            dim=0,
            dim_size=molecule_formula.shape[0],
            reduce="sum",
        ).clamp_min(1.0)
        fragment_normalization_count = molecule_fragment_count[fragment_batch]
        fragment_weight = (
            fragment_ratio * fragment_active / fragment_normalization_count)
        return (molecule_formula, fragment_formula, fragment_ratio,
                fragment_active, fragment_weight,
                fragment_normalization_count)

    def _bound_residual(self, raw_residual):
        scaled = raw_residual / self.residual_bound
        safe_scaled = torch.where(
            scaled.abs() > 1.0e-6, scaled, torch.ones_like(scaled))
        ratio = torch.tanh(safe_scaled) / safe_scaled
        ratio = torch.where(
            scaled.abs() > 1.0e-6,
            ratio,
            1.0 - scaled.square() / 3.0,
        )
        return self.residual_bound * torch.tanh(scaled), ratio

    def _pair_features(self, pair_fragment, fragment_weight,
                       fragment_formula, fragment_batch, molecule_formula,
                       formula_count):
        weighted = pair_fragment * fragment_weight.unsqueeze(-1)
        formula_sum = scatter(
            weighted,
            fragment_formula,
            dim=0,
            dim_size=formula_count,
            reduce="sum",
        )
        formula_square_sum = scatter(
            weighted.square(),
            fragment_formula,
            dim=0,
            dim_size=formula_count,
            reduce="sum",
        )
        pair_features = 0.5 * (formula_sum.square() - formula_square_sum)
        if self.cross_molecule_only:
            molecule_count = molecule_formula.shape[0]
            molecule_sum = scatter(
                weighted,
                fragment_batch,
                dim=0,
                dim_size=molecule_count,
                reduce="sum",
            )
            molecule_square_sum = scatter(
                weighted.square(),
                fragment_batch,
                dim=0,
                dim_size=molecule_count,
                reduce="sum",
            )
            internal_pairs = 0.5 * (
                molecule_sum.square() - molecule_square_sum)
            pair_features = pair_features - scatter(
                internal_pairs,
                molecule_formula,
                dim=0,
                dim_size=formula_count,
                reduce="sum",
            )
        return pair_features

    def _top_pair_diagnostics(self, pair_fragment, pair_condition,
                              fragment_weight, fragment_formula,
                              fragment_batch, formula_count,
                              attribution_scale):
        device = pair_fragment.device
        index_parts = []
        formula_parts = []
        effect_parts = []
        if self.diagnostic_top_k_pairs == 0:
            return {
                "pair_indices": torch.empty((0, 2), dtype=torch.long,
                                             device=device),
                "pair_formula_index": torch.empty((0, ), dtype=torch.long,
                                                   device=device),
                "pair_effect": pair_fragment.new_empty((0, )),
            }
        for formula_index in range(formula_count):
            indices = torch.nonzero(
                (fragment_formula == formula_index)
                & (fragment_weight != 0), as_tuple=False).reshape(-1)
            if indices.numel() < 2:
                continue
            local_pairs = torch.triu_indices(
                indices.numel(), indices.numel(), offset=1, device=device)
            left = indices[local_pairs[0]]
            right = indices[local_pairs[1]]
            if self.cross_molecule_only:
                keep = fragment_batch[left] != fragment_batch[right]
                left, right = left[keep], right[keep]
            if left.numel() == 0:
                continue
            effects = (
                pair_fragment[left] * pair_fragment[right]
                * pair_condition[formula_index]
            ).sum(dim=-1)
            effects = effects * (
                self.interaction_scale / math.sqrt(self.pair_rank)
                * fragment_weight[left] * fragment_weight[right])
            effects = effects * attribution_scale[formula_index]
            if self.diagnostic_top_k_pairs == -1:
                keep_count = effects.numel()
                keep = torch.arange(keep_count, device=device)
            else:
                keep_count = min(self.diagnostic_top_k_pairs, effects.numel())
                keep = torch.topk(effects.abs(), keep_count).indices
            index_parts.append(torch.stack([left[keep], right[keep]], dim=-1))
            formula_parts.append(torch.full(
                (keep_count, ), formula_index, dtype=torch.long,
                device=device))
            effect_parts.append(effects[keep])
        if not index_parts:
            return {
                "pair_indices": torch.empty((0, 2), dtype=torch.long,
                                             device=device),
                "pair_formula_index": torch.empty((0, ), dtype=torch.long,
                                                   device=device),
                "pair_effect": pair_fragment.new_empty((0, )),
            }
        return {
            "pair_indices": torch.cat(index_parts, dim=0),
            "pair_formula_index": torch.cat(formula_parts, dim=0),
            "pair_effect": torch.cat(effect_parts, dim=0),
        }

    def forward(self, data, fragment_h, fragment_batch, condition_h,
                fragment_active=None):
        if fragment_h.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected fragment dim {self.input_dim}, got "
                f"{fragment_h.shape[-1]}")
        if condition_h.shape[-1] != self.condition_input_dim:
            raise ValueError(
                f"expected condition dim {self.condition_input_dim}, got "
                f"{condition_h.shape[-1]}")
        formula_count = condition_h.shape[0]
        (molecule_formula, fragment_formula, fragment_ratio,
         fragment_active, fragment_weight,
         fragment_normalization_count) = self._assignments(
             data, fragment_batch, fragment_active, fragment_h.dtype)

        encoded_fragment = self.fragment_encoder(fragment_h)
        encoded_condition = self.condition_encoder(condition_h)
        main_fragment = self.main_fragment(encoded_fragment)
        pair_fragment = self.pair_fragment(encoded_fragment)
        pair_features = self._pair_features(
            pair_fragment,
            fragment_weight,
            fragment_formula,
            fragment_batch,
            molecule_formula,
            formula_count,
        )

        task_outputs = {}
        for task in self.task_names:
            main_condition = self.main_condition[task](encoded_condition)
            pair_condition = self.pair_condition[task](encoded_condition)
            raw_main_effect = (
                main_fragment * main_condition[fragment_formula]
            ).sum(dim=-1)
            raw_main_effect = raw_main_effect * (
                self.main_scale / math.sqrt(self.main_rank)
                * fragment_weight)
            raw_main_sum = scatter(
                raw_main_effect,
                fragment_formula,
                dim=0,
                dim_size=formula_count,
                reduce="sum",
            )
            raw_interaction_sum = (
                pair_features * pair_condition
            ).sum(dim=-1) * (
                self.interaction_scale / math.sqrt(self.pair_rank))
            raw_residual = raw_main_sum + raw_interaction_sum
            residual, attribution_scale = self._bound_residual(raw_residual)

            # Apply one formula-level scale to every constituent attribution.
            # This keeps main + pair == bounded residual exactly, while the
            # signs and relative shares of all raw effects are unchanged.
            main_effect = (
                raw_main_effect * attribution_scale[fragment_formula])
            main_sum = raw_main_sum * attribution_scale
            interaction_sum = raw_interaction_sum * attribution_scale
            task_output = {
                "raw_main_effect": raw_main_effect,
                "raw_main_sum": raw_main_sum,
                "raw_interaction_sum": raw_interaction_sum,
                "raw_residual": raw_residual,
                "main_effect": main_effect,
                "main_sum": main_sum,
                "interaction_sum": interaction_sum,
                "residual": residual,
                "attribution_scale": attribution_scale,
                "residual_bound": residual.new_full(
                    residual.shape, self.residual_bound),
            }
            if not self.training:
                task_output.update(self._top_pair_diagnostics(
                    pair_fragment,
                    pair_condition,
                    fragment_weight,
                    fragment_formula,
                    fragment_batch,
                    formula_count,
                    attribution_scale,
                ))
            task_outputs[task] = task_output

        return {
            "tasks": task_outputs,
            "fragment_formula_index": fragment_formula,
            "fragment_molecule_index": fragment_batch,
            "fragment_molar_ratio": fragment_ratio,
            "fragment_active": fragment_active,
            "fragment_weight": fragment_weight,
            "fragment_normalization_count": fragment_normalization_count,
        }


class ConditionAwareFragmentExplainer(ConditionAwareFragmentContribution):
    """Conditioned, additive fragment explanation without prediction updates.

    ``condition_reference`` captures the conductivity expected from salt,
    concentration and temperature alone.  Fragment main effects and
    cross-molecule interactions then explain the composition-dependent offset
    from that reference.  The caller decides whether this explanatory score is
    used for prediction; :class:`MolMix` deliberately keeps it separate from
    the frozen Bamboo prediction.
    """

    DEFAULT_TASKS = ("conductivity_explain", )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_reference = torch.nn.ModuleDict({
            task: torch.nn.Linear(self.condition_input_dim, 1)
            for task in self.task_names
        })
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "condition_reference"):
            # A zero reference makes the first explanatory score neutral while
            # retaining gradients for its condition-dependent baseline.
            for projection in self.condition_reference.values():
                torch.nn.init.zeros_(projection.weight)
                torch.nn.init.zeros_(projection.bias)

    def forward(self, data, fragment_h, fragment_batch, condition_h,
                fragment_active=None):
        output = super().forward(data, fragment_h, fragment_batch,
                                 condition_h, fragment_active)
        for task in self.task_names:
            task_output = output["tasks"][task]
            reference = self.condition_reference[task](condition_h).squeeze(-1)
            task_output["condition_reference"] = reference
            task_output["explanation_prediction"] = (
                reference + task_output["residual"])
        return output


class ConditionAwarePropertyFragmentExplainer(
        ConditionAwareFragmentExplainer):
    """Add explicit solvent-property terms to the additive fragment explainer.

    The deployed Bamboo prediction is not changed.  This head models the
    experimental conductivity decomposition as a condition reference, a
    molar-ratio-weighted solvent-property term, and the existing bounded
    fragment main/pair residual.  Property values come from the frozen
    molecular multitask model, so this is an interpretable *model
    decomposition*, not a claim that the listed properties are independent
    causal mediators.
    """

    def __init__(self,
                 *args,
                 property_names=("dc", "vis", "Tm"),
                 property_centers=None,
                 property_scales=None,
                 property_effect_scale: float = 0.10,
                 property_bound: float = 0.25,
                 **kwargs):
        self.property_names = tuple(property_names)
        if not self.property_names:
            raise ValueError("at least one solvent property is required")
        if len(set(self.property_names)) != len(self.property_names):
            raise ValueError("property_names must be unique")
        if property_effect_scale <= 0 or property_bound <= 0:
            raise ValueError("property contribution scales must be positive")
        property_count = len(self.property_names)
        centers = ([0.5] * property_count if property_centers is None else
                   list(property_centers))
        scales = ([0.25] * property_count if property_scales is None else
                  list(property_scales))
        if len(centers) != property_count or len(scales) != property_count:
            raise ValueError(
                "property_centers and property_scales must match property_names")
        if any(float(scale) <= 0 for scale in scales):
            raise ValueError("property_scales must be positive")
        self.property_effect_scale = float(property_effect_scale)
        self.property_bound = float(property_bound)
        super().__init__(*args, **kwargs)
        self.register_buffer("property_centers",
                             torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("property_scales",
                             torch.tensor(scales, dtype=torch.float32))
        self.property_condition = torch.nn.ModuleDict({
            task: torch.nn.Linear(self.condition_input_dim, property_count)
            for task in self.task_names
        })
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "property_condition"):
            # Start from the previous condition + fragment explanation.  The
            # property term then grows only when it improves experimental fit.
            for projection in self.property_condition.values():
                torch.nn.init.zeros_(projection.weight)
                torch.nn.init.zeros_(projection.bias)

    @staticmethod
    def _molecule_formula_index(data, device):
        molecule_counts = data.counts_mol[:, 0]
        return torch.arange(molecule_counts.shape[0], dtype=torch.long,
                            device=device).repeat_interleave(
                                molecule_counts.long())

    def _property_effects(self, data, solvent_properties, condition_h):
        if solvent_properties.ndim != 2:
            raise ValueError("solvent_properties must have shape [molecule, property]")
        property_count = len(self.property_names)
        if solvent_properties.shape[-1] != property_count:
            raise ValueError(
                f"expected {property_count} solvent properties, got "
                f"{solvent_properties.shape[-1]}")
        molecule_formula = self._molecule_formula_index(
            data, solvent_properties.device)
        if solvent_properties.shape[0] != molecule_formula.shape[0]:
            raise ValueError(
                "solvent property count does not match solvent molecule count")
        formula_count = condition_h.shape[0]
        molecule_ratio = data.molar_ratios.to(
            device=solvent_properties.device,
            dtype=solvent_properties.dtype).reshape(-1)
        if molecule_ratio.shape[0] != molecule_formula.shape[0]:
            raise ValueError("solvent molar-ratio count does not match molecules")
        normalized = ((solvent_properties - self.property_centers.to(
            solvent_properties)) / self.property_scales.to(solvent_properties))
        outputs = {}
        for task in self.task_names:
            coefficients = self.property_condition[task](condition_h)
            raw_molecule_effect = (
                normalized * coefficients[molecule_formula]
                * molecule_ratio.unsqueeze(-1) * self.property_effect_scale)
            raw_by_property = scatter(
                raw_molecule_effect,
                molecule_formula,
                dim=0,
                dim_size=formula_count,
                reduce="sum",
            )
            raw_sum = raw_by_property.sum(dim=-1)
            bounded_sum, scale = self._bound_property_sum(raw_sum)
            molecule_effect = raw_molecule_effect * scale[molecule_formula].unsqueeze(-1)
            by_property = raw_by_property * scale.unsqueeze(-1)
            outputs[task] = {
                "property_names": self.property_names,
                "property_values": solvent_properties,
                "property_normalized": normalized,
                "property_coefficients": coefficients,
                "raw_property_molecule_effect": raw_molecule_effect,
                "raw_property_by_name": raw_by_property,
                "raw_property_sum": raw_sum,
                "property_molecule_effect": molecule_effect,
                "property_by_name": by_property,
                "property_sum": bounded_sum,
                "property_attribution_scale": scale,
                "property_bound": bounded_sum.new_full(
                    bounded_sum.shape, self.property_bound),
            }
        return molecule_formula, outputs

    def _bound_property_sum(self, raw_sum):
        scaled = raw_sum / self.property_bound
        safe_scaled = torch.where(
            scaled.abs() > 1.0e-6, scaled, torch.ones_like(scaled))
        ratio = torch.tanh(safe_scaled) / safe_scaled
        ratio = torch.where(
            scaled.abs() > 1.0e-6,
            ratio,
            1.0 - scaled.square() / 3.0,
        )
        return self.property_bound * torch.tanh(scaled), ratio

    def forward(self, data, fragment_h, fragment_batch, condition_h,
                solvent_properties, fragment_active=None):
        output = super().forward(data, fragment_h, fragment_batch,
                                 condition_h, fragment_active)
        molecule_formula, property_outputs = self._property_effects(
            data, solvent_properties, condition_h)
        for task in self.task_names:
            task_output = output["tasks"][task]
            property_output = property_outputs[task]
            task_output.update(property_output)
            task_output["explanation_prediction"] = (
                task_output["condition_reference"]
                + property_output["property_sum"]
                + task_output["residual"])
        output["property_molecule_formula_index"] = molecule_formula
        return output
