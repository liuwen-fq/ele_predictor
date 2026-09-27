"""On-demand adapter for the frozen ordinary and explanation predictor heads."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrozenPredictorAdapter:
    """Load the existing fine-tuned oracle once and expose a small UI-safe API."""

    config_path: Path
    checkpoint_path: Path
    device: str | None = None

    def __post_init__(self) -> None:
        import torch
        import yaml
        from formula_design.predictor import MolMix, Mono

        if not self.config_path.is_file():
            raise FileNotFoundError(f"Oracle config does not exist: {self.config_path}")
        for filename in ("pretrain.pt", "optimal.pt"):
            if not (self.checkpoint_path / filename).is_file():
                raise FileNotFoundError(f"Oracle checkpoint is missing {filename}: {self.checkpoint_path}")
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = yaml.safe_load(self.config_path.read_text())
        model_config = copy.deepcopy(config["model"])
        pretrain_config = model_config.pop("pretrain")
        mono = Mono(**pretrain_config["model_config"])
        mono.load_ckpt(str(self.checkpoint_path / "pretrain.pt"))
        self.model = MolMix(mono, **model_config["model"])
        # MolMix.load_ckpt() restores to the device embedded in the checkpoint.
        # The released oracle was saved on CUDA, while a demo can reasonably be
        # run on a CPU laptop, so explicitly remap its tensor storage here.
        checkpoint = torch.load(self.checkpoint_path / "optimal.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(self.device)

    @staticmethod
    def _value(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value.detach().cpu().reshape(-1)[0])
        except AttributeError:
            return float(value)

    def _task_summary(self, task: dict[str, Any] | None) -> dict[str, Any]:
        if not task:
            return {}
        names = list(task.get("property_names", ()))
        property_by_name = task.get("property_by_name")
        values = [] if property_by_name is None else property_by_name.detach().cpu().reshape(1, -1)[0].tolist()
        return {
            "condition_reference": self._value(task.get("condition_reference")),
            "property_sum": self._value(task.get("property_sum")),
            "main_sum": self._value(task.get("main_sum")),
            "interaction_sum": self._value(task.get("interaction_sum")),
            "residual": self._value(task.get("residual")),
            "explanation_prediction": self._value(task.get("explanation_prediction")),
            "property_by_name": {name: float(value) for name, value in zip(names, values)},
        }

    def _conductivity_details(self, graph: Any, diagnostics: dict[str, Any]) -> dict[str, Any]:
        """Make the released explanation tensors safe to render in the UI.

        The model reports fragment effects as a flat tensor.  This method joins
        it back to the formula's fragment metadata and also keeps the solvent
        property terms on the originating molecule.  It deliberately exposes
        values in the explanation head's normalized space; the UI labels that
        clearly instead of implying unvalidated physical units.
        """
        task = (diagnostics.get("explanations") or {}).get("tasks", {}).get(
            "conductivity_explain"
        )
        solvent_diagnostics = diagnostics.get("solvents")
        if not task or not solvent_diagnostics:
            return {}

        from formula_design.fragment import attach_fragment_attention

        annotated = attach_fragment_attention(
            graph.solv_graphs.fragment_metadata,
            solvent_diagnostics["attention"].detach().cpu().tolist(),
            bool(solvent_diagnostics["neutral_only"]),
        )[0]
        main_effects = task["main_effect"].detach().cpu().reshape(-1).tolist()
        fragment_rows: list[dict[str, Any]] = []
        fragment_lookup: dict[int, dict[str, Any]] = {}
        offset = 0
        for molecule_index, molecule in enumerate(annotated):
            for fragment in molecule["fragments"]:
                if offset >= len(main_effects):
                    raise ValueError("fragment explanation is shorter than fragment metadata")
                row = {
                    "molecule_index": molecule_index,
                    "molecule_name": molecule["molecule_name"],
                    "molecule_smiles": molecule["mapped_smiles"],
                    "fragment_id": int(fragment["fragment_id"]),
                    "fragment_smiles": fragment["smiles"],
                    "pooling_attention": float(fragment["pooling_attention"]),
                    "main_effect_normalized": float(main_effects[offset]),
                }
                fragment_rows.append(row)
                fragment_lookup[offset] = row
                offset += 1
        if offset != len(main_effects):
            raise ValueError("fragment explanation is longer than fragment metadata")

        pairs = []
        pair_indices = task.get("pair_indices")
        if pair_indices is not None:
            pair_effects = task["pair_effect"].detach().cpu().reshape(-1).tolist()
            for (left, right), effect in zip(pair_indices.detach().cpu().tolist(), pair_effects):
                left_row, right_row = fragment_lookup[int(left)], fragment_lookup[int(right)]
                pairs.append({
                    "left": f"{left_row['molecule_name']} · {left_row['fragment_smiles']}",
                    "right": f"{right_row['molecule_name']} · {right_row['fragment_smiles']}",
                    "effect_normalized": float(effect),
                })
        pairs.sort(key=lambda row: abs(row["effect_normalized"]), reverse=True)

        property_names = list(task.get("property_names", ()))
        property_values = task.get("property_values")
        property_effects = task.get("property_molecule_effect")
        molar_ratios = graph.solv_graphs.molar_ratios.detach().cpu().reshape(-1).tolist()
        molecule_property_rows = []
        if property_values is not None and property_effects is not None:
            values = property_values.detach().cpu().tolist()
            effects = property_effects.detach().cpu().tolist()
            for molecule_index, (molecule, values_for_molecule, effects_for_molecule) in enumerate(
                zip(annotated, values, effects)
            ):
                row: dict[str, Any] = {
                    "molecule_index": molecule_index,
                    "molecule_name": molecule["molecule_name"],
                    "molecule_smiles": molecule["mapped_smiles"],
                    "molar_ratio": float(molar_ratios[molecule_index]),
                }
                for name, value, effect in zip(property_names, values_for_molecule, effects_for_molecule):
                    row[f"{name}_value_normalized"] = float(value)
                    row[f"{name}_effect_normalized"] = float(effect)
                row["total_property_effect_normalized"] = float(sum(effects_for_molecule))
                molecule_property_rows.append(row)

        return {
            "fragment_rows": fragment_rows,
            "top_fragment_pairs": pairs[:20],
            "molecule_property_rows": molecule_property_rows,
        }

    def predict(
        self,
        solvents: list[dict[str, Any]],
        salts: list[dict[str, Any]],
        temperature_c: float,
        salt_molar_ratio: float,
    ) -> dict[str, Any]:
        """Predict one formulation. Ratios are normalized within each component group."""
        from formula_design.data import FormulaData, collate_data
        from formula_design.mol import Molecule

        if not solvents or not salts:
            raise ValueError("At least one solvent and one salt are required.")
        def normalize(rows: list[dict[str, Any]]) -> list[float]:
            raw = [max(0.0, float(row["ratio"])) for row in rows]
            total = sum(raw)
            if total <= 0:
                raise ValueError("At least one component ratio must be positive.")
            return [value / total for value in raw]

        solvent_ratios, salt_ratios = normalize(solvents), normalize(salts)
        formula = FormulaData(
            [row["name"] for row in solvents],
            [Molecule.from_smiles(row["smiles"]).get_mapped_smiles() for row in solvents],
            [row["name"] for row in salts],
            [Molecule.from_smiles(row["smiles"]).get_mapped_smiles() for row in salts],
            solvent_ratios,
            salt_ratios,
            temperature=(float(temperature_c) + 273.15) / 373.15,
            concentration=float(salt_molar_ratio),
        )
        graph = collate_data([formula]).to(self.device)
        with self._torch.no_grad():
            prediction = self.model(graph)
        conductivity = self._value(prediction["conductivity"])
        conductivity_ms_cm = float(self._torch.exp(
            self._torch.tensor(conductivity * 8.3344 - 4.6002)
        ).item())
        diagnostics = prediction.get("fragment_diagnostics", {})
        explanations = diagnostics.get("explanations") or prediction.get("fragment_explanations") or {}
        tasks = explanations.get("tasks", {})
        conductivity_explain = self._task_summary(tasks.get("conductivity_explain"))
        if conductivity_explain:
            conductivity_explain["physical_ms_cm"] = float(self._torch.exp(
                self._torch.tensor(conductivity_explain["explanation_prediction"] * 8.3344 - 4.6002)
            ).item())
        try:
            conductivity_details = self._conductivity_details(graph, diagnostics)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            conductivity_details = {"rendering_warning": str(error)}
        return {
            "conductivity_ms_cm": conductivity_ms_cm,
            "anion_ratio": self._value(prediction.get("anion_ratio")),
            "conductivity_explain": conductivity_explain,
            "conductivity_explain_details": conductivity_details,
            "anion_ratio_explain": self._task_summary(tasks.get("anion_ratio_explain")),
            "device": self.device,
            "normalized_solvent_ratios": solvent_ratios,
            "normalized_salt_ratios": salt_ratios,
        }
