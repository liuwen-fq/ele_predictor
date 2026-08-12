# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from formula_design.data import FormulaData, GraphData, MXDataset, collate_data
from formula_design.fragment import attach_fragment_attention
from formula_design.predictor import MolMix, Mono
from formula_design.train import FormulaTrainer, TrainConfig
from formula_design.utils import setup_default_logging, to_dense_batch

logger = setup_default_logging()

outputs = {}
LOG_CONDUCTIVITY_SCALE = 8.3344


def make_hook(output_store, layer_name):

    def hook_fn(module, input, output):
        # Convert to list for JSON serialization
        output_store["layer_outputs"][layer_name] = output.detach().cpu(
        ).tolist()

    return hook_fn


def collect_fragment_attention(component_graph, diagnostics):
    """Match last-forward fragment attention with serialized molecule metadata."""

    if diagnostics is None:
        return None
    if "fragment_metadata" not in component_graph:
        raise ValueError(
            "fragment metadata is missing; reprocess the prediction dataset")
    return attach_fragment_attention(
        component_graph.fragment_metadata,
        diagnostics["attention"].detach().cpu().tolist(),
        diagnostics["neutral_only"],
    )


def collect_fragment_gates(diagnostics):
    """Serialize molecular and formula-level Fragment sidecar gates."""

    if diagnostics is None:
        return None
    gates = {}
    for component in ("solvents", "salts"):
        component_diagnostics = diagnostics.get(component)
        if (component_diagnostics is not None
                and component_diagnostics.get("gate") is not None):
            gates[f"molecule_{component}"] = float(
                component_diagnostics["gate"].detach().cpu())
    formula_gates = diagnostics.get("formula_gates")
    if formula_gates is not None:
        gates.update({
            f"formula_{name}": float(value.detach().cpu())
            for name, value in formula_gates.items()
        })
    return gates or None


def collect_fragment_contributions(component_graph, annotated_metadata,
                                   diagnostics,
                                   task="conductivity_mistry",
                                   diagnostic_key="contributions"):
    """Attach signed main effects and top pair interactions to fragments."""

    if diagnostics is None or diagnostics.get(diagnostic_key) is None:
        return None
    contribution_data = diagnostics[diagnostic_key]
    task_data = contribution_data["tasks"][task]
    main_effects = task_data["main_effect"].detach().cpu().tolist()
    fragment_lookup = {}
    offset = 0
    for formula_index, formula in enumerate(annotated_metadata):
        for molecule_index, molecule in enumerate(formula):
            for fragment in molecule["fragments"]:
                if offset >= len(main_effects):
                    raise ValueError(
                        "fragment contribution is shorter than metadata")
                fragment["conductivity_main_effect_normalized"] = float(
                    main_effects[offset])
                fragment["conductivity_main_effect_delta_ln"] = float(
                    main_effects[offset] * LOG_CONDUCTIVITY_SCALE)
                fragment_lookup[offset] = (
                    formula_index,
                    molecule_index,
                    int(fragment["fragment_id"]),
                    molecule["molecule_name"],
                    fragment.get("smiles", ""),
                )
                offset += 1
    if offset != len(main_effects):
        raise ValueError("fragment contribution is longer than metadata")

    formula_count = len(annotated_metadata)
    raw_main_sums = task_data.get(
        "raw_main_sum", task_data["main_sum"]).detach().cpu().tolist()
    raw_interaction_sums = task_data.get(
        "raw_interaction_sum",
        task_data["interaction_sum"]).detach().cpu().tolist()
    raw_residuals = task_data.get(
        "raw_residual", task_data["residual"]).detach().cpu().tolist()
    attribution_scales = task_data.get(
        "attribution_scale",
        torch.ones_like(task_data["residual"])).detach().cpu().tolist()
    residual_bounds = task_data.get(
        "residual_bound",
        torch.full_like(task_data["residual"], float("nan")),
    ).detach().cpu().tolist()
    condition_references = task_data.get(
        "condition_reference",
        torch.zeros_like(task_data["residual"]),
    ).detach().cpu().tolist()
    explanation_predictions = task_data.get(
        "explanation_prediction",
        task_data["residual"],
    ).detach().cpu().tolist()
    property_names = list(task_data.get("property_names", ()))
    property_sums = task_data.get(
        "property_sum", torch.zeros_like(task_data["residual"])
    ).detach().cpu().tolist()
    property_by_name = task_data.get("property_by_name")
    if property_by_name is None:
        property_by_name = torch.empty(
            (len(property_sums), 0), device=task_data["residual"].device)
    property_by_name = property_by_name.detach().cpu().tolist()
    output = [{
        "task": task,
        "raw_main_effect_sum_normalized": float(raw_main),
        "raw_interaction_sum_normalized": float(raw_interaction),
        "raw_total_residual_normalized": float(raw_residual),
        "main_effect_sum_normalized": float(value),
        "interaction_sum_normalized": float(interaction),
        "total_residual_normalized": float(residual),
        "attribution_scale": float(scale),
        "residual_bound_normalized": float(bound),
        "condition_reference_normalized": float(reference),
        "property_sum_normalized": float(property_sum),
        "property_sum_by_name_normalized": {
            name: float(value)
            for name, value in zip(property_names, by_name)
        },
        "total_composition_offset_normalized": float(property_sum + residual),
        "explanation_prediction_normalized": float(explanation),
        "total_residual_delta_ln": float(
            residual * LOG_CONDUCTIVITY_SCALE),
        "fragment_conductivity_multiplier": float(math.exp(
            max(-50.0, min(50.0, residual * LOG_CONDUCTIVITY_SCALE)))),
        "conductivity_multiplier": float(math.exp(max(
            -50.0,
            min(50.0, (property_sum + residual) * LOG_CONDUCTIVITY_SCALE)))),
        "top_fragment_pairs": [],
    } for (raw_main, raw_interaction, raw_residual, value, interaction,
           residual, scale, bound, reference, property_sum, by_name,
           explanation) in zip(
        raw_main_sums,
        raw_interaction_sums,
        raw_residuals,
        task_data["main_sum"].detach().cpu().tolist(),
        task_data["interaction_sum"].detach().cpu().tolist(),
        task_data["residual"].detach().cpu().tolist(),
        attribution_scales,
        residual_bounds,
        condition_references,
        property_sums,
        property_by_name,
        explanation_predictions,
    )]
    if len(output) != formula_count:
        raise ValueError("formula contribution count does not match metadata")

    pair_indices = task_data.get("pair_indices")
    if pair_indices is None:
        return output
    pair_indices = pair_indices.detach().cpu().tolist()
    pair_formulas = task_data["pair_formula_index"].detach().cpu().tolist()
    pair_effects = task_data["pair_effect"].detach().cpu().tolist()
    for (left, right), formula_index, effect in zip(
            pair_indices, pair_formulas, pair_effects):
        left_info = fragment_lookup[int(left)]
        right_info = fragment_lookup[int(right)]
        if left_info[0] != formula_index or right_info[0] != formula_index:
            raise ValueError("pair contribution formula assignment mismatch")
        output[formula_index]["top_fragment_pairs"].append({
            "effect_normalized": float(effect),
            "effect_delta_ln": float(effect * LOG_CONDUCTIVITY_SCALE),
            "left": {
                "molecule_index": left_info[1],
                "molecule_name": left_info[3],
                "fragment_id": left_info[2],
                "fragment_smiles": left_info[4],
            },
            "right": {
                "molecule_index": right_info[1],
                "molecule_name": right_info[3],
                "fragment_id": right_info[2],
                "fragment_smiles": right_info[4],
            },
        })
    return output


def collect_property_explanations(annotated_metadata, diagnostics,
                                  molar_ratios=None,
                                  task="conductivity_explain"):
    """Serialize per-solvent, signed property terms from the joint head."""

    if diagnostics is None or diagnostics.get("property_explanations") is None:
        return None
    contribution_data = diagnostics["property_explanations"]
    task_data = contribution_data["tasks"][task]
    property_names = list(task_data.get("property_names", ()))
    if not property_names:
        return None
    values = task_data["property_values"].detach().cpu().tolist()
    normalized = task_data["property_normalized"].detach().cpu().tolist()
    effects = task_data["property_molecule_effect"].detach().cpu().tolist()
    if molar_ratios is None:
        molar_ratios = [0.0] * len(values)
    else:
        molar_ratios = molar_ratios.detach().cpu().reshape(-1).tolist()
    if len(molar_ratios) != len(values):
        raise ValueError("property molar-ratio count does not match molecules")
    formula_count = len(annotated_metadata)
    output = [{
        "task": task,
        "property_names": property_names,
        "raw_property_sum_normalized": float(raw_sum),
        "property_sum_normalized": float(property_sum),
        "property_sum_by_name_normalized": {
            name: float(value)
            for name, value in zip(property_names, by_name)
        },
        "property_attribution_scale": float(scale),
        "property_bound_normalized": float(bound),
        "solvent_property_contributions": [],
    } for raw_sum, property_sum, by_name, scale, bound in zip(
        task_data["raw_property_sum"].detach().cpu().tolist(),
        task_data["property_sum"].detach().cpu().tolist(),
        task_data["property_by_name"].detach().cpu().tolist(),
        task_data["property_attribution_scale"].detach().cpu().tolist(),
        task_data["property_bound"].detach().cpu().tolist(),
    )]
    if len(output) != formula_count:
        raise ValueError("formula property contribution count does not match metadata")
    offset = 0
    for formula_index, formula in enumerate(annotated_metadata):
        for molecule_index, molecule in enumerate(formula):
            if offset >= len(values):
                raise ValueError("property contribution is shorter than metadata")
            output[formula_index]["solvent_property_contributions"].append({
                "molecule_index": molecule_index,
                "molecule_name": molecule.get("molecule_name", "unknown"),
                "molecule_smiles": molecule.get("mapped_smiles", ""),
                "molar_ratio": float(molar_ratios[offset]),
                "property_values_normalized": {
                    name: float(value)
                    for name, value in zip(property_names, values[offset])
                },
                "property_values_centered_scaled": {
                    name: float(value)
                    for name, value in zip(property_names, normalized[offset])
                },
                "property_effects_normalized": {
                    name: float(value)
                    for name, value in zip(property_names, effects[offset])
                },
                "total_property_effect_normalized": float(sum(effects[offset])),
            })
            offset += 1
    if offset != len(values):
        raise ValueError("property contribution is longer than metadata")
    return output


def collect_molecular_property_profiles(annotated_metadata, diagnostics,
                                        molar_ratios=None):
    """Serialize all frozen molecular-property predictions per solvent.

    The three properties used by the explanation head (``dc``, ``vis`` and
    ``Tm``) remain separately attributed in ``property_explanations``.  This
    diagnostic export intentionally includes every multitask property so that
    the report can describe supporting trends such as density, surface tension
    or volatility without falsely presenting them as trained conductivity
    contribution terms.
    """

    if diagnostics is None:
        return None
    property_data = diagnostics.get("solvent_properties")
    if not property_data:
        return None
    property_names = sorted(property_data)
    values = {
        name: tensor.detach().cpu().reshape(-1).tolist()
        for name, tensor in property_data.items()
    }
    molecule_count = len(next(iter(values.values())))
    if any(len(items) != molecule_count for items in values.values()):
        raise ValueError("molecular property lengths do not match")
    if molar_ratios is None:
        molar_ratios = [0.0] * molecule_count
    else:
        molar_ratios = molar_ratios.detach().cpu().reshape(-1).tolist()
    if len(molar_ratios) != molecule_count:
        raise ValueError("molecular-property molar-ratio count does not match")

    output = [[] for _ in annotated_metadata]
    offset = 0
    for formula_index, formula in enumerate(annotated_metadata):
        for molecule_index, molecule in enumerate(formula):
            if offset >= molecule_count:
                raise ValueError("molecular property profile is shorter than metadata")
            output[formula_index].append({
                "molecule_index": molecule_index,
                "molecule_name": molecule.get("molecule_name", "unknown"),
                "molecule_smiles": molecule.get("mapped_smiles", ""),
                "molar_ratio": float(molar_ratios[offset]),
                "property_values_normalized": {
                    name: float(values[name][offset]) for name in property_names
                },
            })
            offset += 1
    if offset != molecule_count:
        raise ValueError("molecular property profile is longer than metadata")
    return output


def main():

    parser = argparse.ArgumentParser(description='test local')
    parser.add_argument('--conf', type=str, default='test_config.yaml')
    args = parser.parse_args()

    assert os.path.exists(args.conf) and args.conf.endswith(
        '.yaml'), f'yaml config {args.conf} not found.'

    with open(args.conf, 'r') as f:
        test_conf = yaml.safe_load(f)

    output_path = test_conf.pop("output_path")
    ckpt_path = test_conf.pop("ckpt_path")
    model_config = test_conf.pop("model")
    pretrain_config = model_config.pop("pretrain")
    pretrained_model = Mono(**pretrain_config["model_config"])
    pretrained_model.load_ckpt(os.path.join(ckpt_path, "pretrain.pt"))
    model = MolMix(pretrained_model, **model_config["model"])
    model.load_ckpt(os.path.join(ckpt_path, "optimal.pt"))
    model.to("cuda")
    model.eval()

    dataset = MXDataset(test_conf)
    valid_dl = DataLoader(dataset,
                          batch_size=256,
                          shuffle=False,
                          drop_last=False,
                          collate_fn=collate_data)

    conductivities_pred = []
    conductivities_base_pred = []
    conductivities_explain_pred = []
    anion_ratios_pred = []
    n1s = []
    n2s = []
    T0s = []
    As = []
    Bs = []
    sigma0s = []
    viss = []
    fragment_attention = []
    fragment_contributions = []
    fragment_explanations = []
    property_explanations = []
    molecular_property_profiles = []
    fragment_gates = None

    for batch_idx, graph in enumerate(valid_dl):

        graph: GraphData = graph.to("cuda")

        batch_outputs = {"layer_outputs": {}}
        hook1 = model.readout_pred.sigma.register_forward_hook(
            make_hook(batch_outputs, "readout_pred.sigma"))
        hook2 = model.readout_pred.n1.register_forward_hook(
            make_hook(batch_outputs, "readout_pred.n1"))
        hook3 = model.readout_pred.n2.register_forward_hook(
            make_hook(batch_outputs, "readout_pred.n2"))
        hook4 = model.readout_pred.T0.register_forward_hook(
            make_hook(batch_outputs, "readout_pred.T0"))
        hook5 = model.readout_pred.A.register_forward_hook(
            make_hook(batch_outputs, "readout_pred.A"))
        hook6 = model.readout_pred.B.register_forward_hook(
            make_hook(batch_outputs, "readout_pred.B"))
        hook7 = model.aggr_vis.register_forward_hook(
            make_hook(batch_outputs, "aggr_vis"))

        pred = model(graph)

        hook1.remove()
        hook2.remove()
        hook3.remove()
        hook4.remove()
        hook5.remove()
        hook6.remove()
        hook7.remove()

        pred_conductivities = torch.exp(pred["conductivity"].squeeze(-1) *
                                        8.3344 - 4.6002)
        pred_base_conductivities = torch.exp(
            pred["conductivity_base"].squeeze(-1) * 8.3344 - 4.6002)
        pred_explain_conductivities = None
        if "conductivity_explain" in pred:
            pred_explain_conductivities = torch.exp(
                pred["conductivity_explain"].squeeze(-1) * 8.3344 - 4.6002)
        pred_anion_ratios = pred["anion_ratio"].squeeze(-1)

        fragment_diagnostics = pred["fragment_diagnostics"]
        fragment_gates = collect_fragment_gates(fragment_diagnostics)
        solvents_attention = collect_fragment_attention(
            graph.solv_graphs, fragment_diagnostics["solvents"])
        salts_attention = collect_fragment_attention(
            graph.salt_graphs, fragment_diagnostics["salts"])
        if solvents_attention is None or salts_attention is None:
            fragment_attention.extend([None] * pred_conductivities.shape[0])
            fragment_contributions.extend(
                [None] * pred_conductivities.shape[0])
            fragment_explanations.extend(
                [None] * pred_conductivities.shape[0])
            property_explanations.extend(
                [None] * pred_conductivities.shape[0])
            molecular_property_profiles.extend(
                [None] * pred_conductivities.shape[0])
        else:
            batch_contributions = collect_fragment_contributions(
                graph.solv_graphs,
                solvents_attention,
                fragment_diagnostics,
            )
            fragment_attention.extend({
                "solvents": solvents,
                "salts": salts,
            } for solvents, salts in zip(solvents_attention, salts_attention))
            if batch_contributions is None:
                fragment_contributions.extend(
                    [None] * pred_conductivities.shape[0])
            else:
                fragment_contributions.extend(batch_contributions)
            batch_explanations = collect_fragment_contributions(
                graph.solv_graphs,
                solvents_attention,
                fragment_diagnostics,
                task="conductivity_explain",
                diagnostic_key="explanations",
            )
            if batch_explanations is None:
                fragment_explanations.extend(
                    [None] * pred_conductivities.shape[0])
            else:
                fragment_explanations.extend(batch_explanations)
            batch_property_explanations = collect_property_explanations(
                solvents_attention,
                fragment_diagnostics,
                graph.solv_graphs.molar_ratios,
            )
            if batch_property_explanations is None:
                property_explanations.extend(
                    [None] * pred_conductivities.shape[0])
            else:
                property_explanations.extend(batch_property_explanations)
            batch_property_profiles = collect_molecular_property_profiles(
                solvents_attention,
                fragment_diagnostics,
                graph.solv_graphs.molar_ratios,
            )
            if batch_property_profiles is None:
                molecular_property_profiles.extend(
                    [None] * pred_conductivities.shape[0])
            else:
                molecular_property_profiles.extend(batch_property_profiles)

        for c in pred_conductivities:
            conductivities_pred.append(c.item())
        for c in pred_base_conductivities:
            conductivities_base_pred.append(c.item())
        if pred_explain_conductivities is not None:
            for c in pred_explain_conductivities:
                conductivities_explain_pred.append(c.item())

        for a in pred_anion_ratios:
            anion_ratios_pred.append(a.item())

        for sigma0 in batch_outputs["layer_outputs"]["readout_pred.sigma"]:
            if isinstance(sigma0, (float, int)):
                sigma0s.append(sigma0)
            else:
                sigma0s.append(sigma0[0])
        for n1 in batch_outputs["layer_outputs"]["readout_pred.n1"]:
            if isinstance(n1, (float, int)):
                n1s.append(n1)
            else:
                n1s.append(n1[0])
        for n2 in batch_outputs["layer_outputs"]["readout_pred.n2"]:
            if isinstance(n2, (float, int)):
                n2s.append(n2)
            else:
                n2s.append(n2[0])
        for A in batch_outputs["layer_outputs"]["readout_pred.A"]:
            if isinstance(A, (float, int)):
                As.append(A)
            else:
                As.append(A[0])
        for B in batch_outputs["layer_outputs"]["readout_pred.B"]:
            if isinstance(B, (float, int)):
                Bs.append(B)
            else:
                Bs.append(B[0])
        for T0 in batch_outputs["layer_outputs"]["readout_pred.T0"]:
            if isinstance(T0, (float, int)):
                T0s.append(T0)
            else:
                T0s.append(T0[0])

        if isinstance(batch_outputs["layer_outputs"]["aggr_vis"],
                      (float, int)):
            viss.append(batch_outputs["layer_outputs"]["aggr_vis"])
        else:
            for vis in batch_outputs["layer_outputs"]["aggr_vis"]:
                viss.append(vis)

        logger.info(f"Finishing prediction of batch {batch_idx}")

    with open(os.path.join(test_conf["save_dir"], "processed_data.json"),
              "r") as file:
        test_data = json.load(file)

    for i, entry in enumerate(test_data):
        entry["temperature"] = entry["temperature"] * 373.15 - 273.15
        entry["conductivity_pred"] = conductivities_pred[i]
        entry["conductivity_base_pred"] = conductivities_base_pred[i]
        if conductivities_explain_pred:
            entry["conductivity_explain_pred"] = conductivities_explain_pred[i]
        entry["anion_ratio_pred"] = anion_ratios_pred[i]
        entry["sigma0"] = sigma0s[i]
        entry["n1"] = n1s[i]
        entry["n2"] = n2s[i]
        entry["A"] = As[i]
        entry["B"] = Bs[i]
        entry["T0"] = T0s[i]
        entry["vis"] = viss[i]
        if fragment_attention[i] is not None:
            entry["fragment_attention"] = fragment_attention[i]
        if fragment_gates is not None:
            entry["fragment_gates"] = fragment_gates
        if fragment_contributions[i] is not None:
            entry["fragment_contributions"] = fragment_contributions[i]
        if fragment_explanations[i] is not None:
            entry["fragment_explanations"] = fragment_explanations[i]
        if property_explanations[i] is not None:
            entry["property_explanations"] = property_explanations[i]
        if molecular_property_profiles[i] is not None:
            entry["molecular_property_profiles"] = molecular_property_profiles[i]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    main()
