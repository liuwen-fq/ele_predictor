"""Read-only access to the existing evaluation, catalogue, and generation assets.

The demo intentionally reads the project's frozen artifacts.  It never writes
into a training directory and it never loads the multi-gigabyte prediction JSON
at page start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DemoPaths:
    """Locations of the checked-in/run-time assets consumed by the demo."""

    root: Path

    @property
    def conductivity_metrics(self) -> Path:
        return self.root / "run/anion_ratio_property_fragment_explanation/results/conductivity_metrics.json"

    @property
    def conductivity_explain_metrics(self) -> Path:
        return self.root / "run/anion_ratio_property_fragment_explanation/results/conductivity_explain_metrics.json"

    @property
    def anion_metrics(self) -> Path:
        return self.root / "run/anion_ratio_property_fragment_explanation/results/anion_ratio_explain_metrics.json"

    @property
    def gdb17_catalogue(self) -> Path:
        return self.root / "run/generation_gdb17_mechanism_auxiliary_comparison/common/gdb17_electrolyte_property_catalog.json"

    @property
    def gdb17_audit(self) -> Path:
        return self.root / "run/generation_gdb17_mechanism_auxiliary_comparison/common/gdb17_electrolyte_property_catalog_audit.json"

    @property
    def property_constraints(self) -> Path:
        return self.root / "run/property_fragment_attribution_v2/generation_design_constraints/generation_property_constraints.csv"

    @property
    def generated_formulas(self) -> Path:
        return self.root / "run/generation_mechanism_auxiliary_comparison/G0-Q-comp/G0_seed_123/G0-MA/generated/generated_formulas.json"

    @property
    def generated_solvents(self) -> Path:
        return self.root / "run/generation_bamboo_composition_comparison/G2-b/candidate_pool/solvents.json"

    @property
    def generated_salts(self) -> Path:
        return self.root / "run/generation_bamboo_composition_comparison/G2-b/candidate_pool/salts.json"

    @property
    def oracle_config(self) -> Path:
        return self.root / "config/test_results/anion_ratio_property_fragment_predict.yaml"

    @property
    def oracle_checkpoint(self) -> Path:
        return self.root / "run/anion_ratio_property_fragment_explanation/ckpt_formula_ft"

    @property
    def global_fragment_figures(self) -> tuple[Path, Path]:
        """Separate high- and low-conductivity direct-contribution figures."""

        directory = self.root / "run/property_fragment_attribution_v2"
        return (
            directory / "high_conductivity_fragment_direct_contributions.svg",
            directory / "low_conductivity_fragment_direct_contributions.svg",
        )

    @property
    def global_fragment_figure(self) -> Path:
        """Backward-compatible first fragment figure (high conductivity)."""

        return self.global_fragment_figures[0]

    @property
    def global_property_figures(self) -> tuple[Path, Path, Path]:
        """Separate dc, vis, and Tm explanation figures in display order."""

        directory = self.root / "run/property_fragment_attribution_v2"
        return (
            directory / "motif_carrier_dc_terms.svg",
            directory / "motif_carrier_vis_terms.svg",
            directory / "motif_carrier_Tm_terms.svg",
        )

    @property
    def global_property_figure(self) -> Path:
        """Backward-compatible first property figure (dc)."""

        return self.global_property_figures[0]


def repo_root(start: Path | None = None) -> Path:
    """Locate the repository without assuming Streamlit's working directory."""
    current = (start or Path(__file__)).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "formula_design").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root containing formula_design/ and config/.")


def default_paths() -> DemoPaths:
    return DemoPaths(repo_root())


@lru_cache(maxsize=16)
def load_json(path: str) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    return path


def load_metric_rows(paths: DemoPaths) -> list[dict[str, Any]]:
    """Normalize the three existing metric artifacts into table-ready rows."""
    conductivity = load_json(str(require_file(paths.conductivity_metrics, "conductivity metrics")))
    conductivity_explain = load_json(
        str(require_file(paths.conductivity_explain_metrics, "conductivity explanation metrics"))
    )
    anion = load_json(str(require_file(paths.anion_metrics, "anion-ratio metrics")))
    return [
        {
            "task": "Conductivity", "head": "Main", "n": conductivity["num_experimental_samples"],
            "r2": conductivity["conductivity_r2"], "mae": conductivity["conductivity_mae_mS_cm"],
            "rmse": conductivity["conductivity_rmse_mS_cm"], "unit": "mS/cm",
        },
        {
            "task": "Conductivity", "head": "Explain", "n": conductivity_explain["num_experimental_samples"],
            "r2": conductivity_explain["conductivity_r2"],
            "mae": conductivity_explain["conductivity_mae_mS_cm"],
            "rmse": conductivity_explain["conductivity_rmse_mS_cm"], "unit": "mS/cm",
        },
        {
            "task": "Anion ratio", "head": "Main", "n": anion["ordinary"]["n"],
            "r2": anion["ordinary"]["r2"], "mae": anion["ordinary"]["mae"],
            "rmse": anion["ordinary"]["rmse"], "unit": "ratio",
        },
        {
            "task": "Anion ratio", "head": "Explain", "n": anion["explain"]["n"],
            "r2": anion["explain"]["r2"], "mae": anion["explain"]["mae"],
            "rmse": anion["explain"]["rmse"], "unit": "ratio",
        },
    ]


def load_generation_assets(paths: DemoPaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a frozen G2-b sample and the vocabulary required to render BOWs."""
    rows = load_json(str(require_file(paths.generated_formulas, "frozen generator output")))
    solvents = load_json(str(require_file(paths.generated_solvents, "generator solvent vocabulary")))
    salts = load_json(str(require_file(paths.generated_salts, "generator salt vocabulary")))
    return rows, solvents, salts
