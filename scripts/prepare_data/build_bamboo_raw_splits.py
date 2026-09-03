#!/usr/bin/env python3
"""Build reproducible local predictor datasets from Bamboo's raw data_oss.

No serialized Bamboo dataset files are read.  Formula records are split first,
then normalized and graph-preprocessed by this repository's FormulaData code.
The molecule pretraining set is deliberately kept whole.
"""

from __future__ import annotations

import argparse
import copy
import json
import lzma
import math
import random
import shutil
from pathlib import Path

import numpy as np
import yaml

from formula_design.data.dataset import MXDataset, MonoDataset


FORMULA_KEY_MAP = {
    "concentration": "salt_molar_ratio",
    "temperature": "temperature",
    "conductivity": "conductivity",
    "conductivity_mask": "conductivity_mask",
    "anion_ratio": "anion_ratio",
    "anion_ratio_mask": "anion_ratio_mask",
}
MD_KEY_MAP = {
    "concentration": "salt_molar_ratio",
    "temperature": "temperature",
    "conductivity_ne": "conductivity_ne",
    "conductivity_mistry": "conductivity_mistry",
    "anion_ratio": "anion_ratio",
}
MONO_KEY_MAP = {
    "Tm": "Tm", "Tm_mask": "Tm_mask", "bp": "bp", "bp_mask": "bp_mask",
    "nD": "nD", "nD_T": "nD_T", "nD_mask": "nD_mask",
    "nD_liquid": "nD_liquid", "nD_liquid_T": "nD_liquid_T",
    "nD_liquid_mask": "nD_liquid_mask", "pka_a": "pka_a",
    "pka_a_mask": "pka_a_mask", "pka_b": "pka_b",
    "pka_b_mask": "pka_b_mask", "dc": "dc", "dc_T": "dc_T",
    "dc_mask": "dc_mask", "ST": "ST", "ST_T": "ST_T",
    "ST_mask": "ST_mask", "density": "density",
    "density_T": "density_T", "density_mask": "density_mask", "vis": "vis",
    "vis_T": "vis_T", "vis_mask": "vis_mask", "vapP": "vapP",
    "vapP_T": "vapP_T", "vapP_mask": "vapP_mask",
}


def load_xz_json(path: Path) -> list[dict]:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def split_records(records: list[dict], fractions: tuple[float, ...], seed: int) -> list[list[dict]]:
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    boundaries = []
    running = 0
    for fraction in fractions[:-1]:
        running += round(len(records) * fraction)
        boundaries.append(running)
    chunks = []
    start = 0
    for end in (*boundaries, len(records)):
        chunks.append([records[index] for index in indices[start:end]])
        start = end
    if sum(map(len, chunks)) != len(records):
        raise AssertionError("split lost or duplicated records")
    return chunks


def split_finetune_records(md_records: list[dict],
                           experimental_records: list[dict],
                           seed: int) -> list[list[dict]]:
    """Split MD and experimental supervision independently before merging.

    MD records come directly from ``formula_pretrain_md_data`` and experimental
    records from ``formula_finetune_exp_data``.  The former is augmented only
    with the missing mask/interface fields required by FormulaData; its fake
    conductivity is always masked and can never contribute to a loss.
    """
    md_records = copy.deepcopy(md_records)
    for record in md_records:
        if "anion_ratio" not in record:
            raise ValueError("formula_pretrain_md_data record lacks anion_ratio")
        record["anion_ratio_mask"] = True
        # FormulaData has one fixed schema for the mixed fine-tuning loader.
        # This value is deliberately ignored because conductivity_mask is False.
        record["conductivity"] = 1.0
        record["conductivity_mask"] = False
    if any(not bool(record.get("conductivity_mask"))
           or bool(record.get("anion_ratio_mask"))
           for record in experimental_records):
        raise ValueError("formula_finetune_exp_data has unexpected label masks")
    md_splits = split_records(md_records, (0.70, 0.15, 0.15), seed)
    exp_splits = split_records(experimental_records, (0.70, 0.15, 0.15), seed + 1)
    merged = []
    for split_index, (md_split, exp_split) in enumerate(zip(md_splits, exp_splits)):
        records = [*md_split, *exp_split]
        random.Random(seed + 10 + split_index).shuffle(records)
        merged.append(records)
    return merged


def normalize_formula_records(records: list[dict]) -> list[dict]:
    """Apply Bamboo's published formula preprocessing to an isolated split."""
    normalized = copy.deepcopy(records)
    for formula in normalized:
        for component_name in ("solvents", "salts"):
            components = formula[component_name]
            total = sum(component["molar_ratio"] for component in components)
            if total <= 0:
                raise ValueError(f"non-positive {component_name} ratio total")
            for component in components:
                component["molar_ratio"] /= total
        if not 0 <= formula["salt_molar_ratio"] < 1:
            raise ValueError("salt_molar_ratio must be in [0, 1)")
        formula["temperature"] = (formula["temperature"] + 273.15) / 373.15
        for key, value in list(formula.items()):
            if key.startswith("conductivity") and key != "conductivity_mask":
                if value <= 0:
                    raise ValueError(f"{key} must be positive for log normalization")
                formula[key] = (math.log(value) + 4.6002) / 8.3344
    return normalized


def normalize_mono_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Mirror Bamboo mono-pretraining min/max normalization on the full set."""
    normalized = copy.deepcopy(records)
    properties = [key for key in normalized[0]
                  if key not in {"name", "smiles"} and "mask" not in key]
    minima = {key: float("inf") for key in properties}
    maxima = {key: float("-inf") for key in properties}
    for record in normalized:
        for key in properties:
            if record.get(f"{key}_mask", True):
                minima[key] = min(minima[key], record[key])
                maxima[key] = max(maxima[key], record[key])
    for key in properties:
        if "_T" in key:
            minima[key] = 0
    for record in normalized:
        for key in properties:
            width = maxima[key] - minima[key]
            record[key] = 0.5 if np.isclose(width, 0) else (record[key] - minima[key]) / width
    return normalized, {"min_values": minima, "max_values": maxima}


def write_config(directory: Path, data_cls: str, key_map: dict) -> Path:
    config = {
        "data_cls": data_cls,
        "json_path": str(directory / "processed_data.json"),
        "save_dir": str(directory),
        "key_map": key_map,
        "kwargs": {},
    }
    path = directory / "dataset_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def is_current(directory: Path, source: Path | str, split_name: str, seed: int | None) -> bool:
    manifest_path = directory / "split_manifest.json"
    pkl_path = directory / "processed_data.pkl"
    if not manifest_path.is_file() or not pkl_path.is_file() or pkl_path.stat().st_size == 0:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (manifest.get("schema_version") == 2
            and manifest.get("source") == str(source)
            and manifest.get("split") == split_name
            and manifest.get("seed") == seed)


def build_formula_split(directory: Path, records: list[dict], key_map: dict,
                        source: Path | str, split_name: str, seed: int, force: bool) -> None:
    if not force and is_current(directory, source, split_name, seed):
        print(f"Dataset already current: {directory}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    prepared = normalize_formula_records(records)
    (directory / "processed_data.json").write_text(
        json.dumps(prepared, ensure_ascii=False), encoding="utf-8")
    config_path = write_config(directory, "FormulaData", key_map)
    pkl_path = directory / "processed_data.pkl"
    if pkl_path.exists():
        pkl_path.unlink()
    MXDataset.process(str(config_path))
    masks = {
        "conductivity_mask_true": sum(bool(record.get("conductivity_mask"))
                                      for record in records),
        "anion_ratio_mask_true": sum(bool(record.get("anion_ratio_mask"))
                                     for record in records),
    }
    (directory / "split_manifest.json").write_text(json.dumps({
        "schema_version": 2, "source": str(source), "split": split_name, "seed": seed,
        "records": len(records), "label_masks": masks,
        "normalization": "Bamboo formula preprocessing",
    }, indent=2) + "\n", encoding="utf-8")


def build_mono(directory: Path, records: list[dict], source: Path, force: bool) -> None:
    if not force and is_current(directory, source, "full", None):
        print(f"Dataset already current: {directory}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    prepared, min_max = normalize_mono_records(records)
    (directory / "processed_data.json").write_text(
        json.dumps(prepared, ensure_ascii=False), encoding="utf-8")
    (directory / "min_max_values.json").write_text(
        json.dumps(min_max, indent=2) + "\n", encoding="utf-8")
    config_path = write_config(directory, "MonoData", MONO_KEY_MAP)
    pkl_path = directory / "processed_data.pkl"
    if pkl_path.exists():
        pkl_path.unlink()
    MonoDataset.process(str(config_path))
    (directory / "split_manifest.json").write_text(json.dumps({
        "schema_version": 2, "source": str(source), "split": "full", "seed": None,
        "records": len(records),
        "normalization": "Bamboo mono-pretraining min/max preprocessing",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="run/bamboo_raw_data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--datasets", nargs="+", choices=("finetune", "md", "mono"),
                        default=("finetune", "md", "mono"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    if args.force and output_root.exists():
        # Only delete data directories produced by this script, never the root.
        for name in ("ft_train", "ft_valid", "ft_test", "md_train", "md_valid", "mono"):
            candidate = output_root / name
            if candidate.is_dir():
                shutil.rmtree(candidate)

    if "finetune" in args.datasets:
        md_source = data_root / "formula_pretrain_md_data.json.xz"
        exp_source = data_root / "formula_finetune_exp_data.json.xz"
        train, valid, test = split_finetune_records(
            load_xz_json(md_source), load_xz_json(exp_source), args.seed)
        source = f"{md_source}|{exp_source}"
        for name, records in (("ft_train", train), ("ft_valid", valid), ("ft_test", test)):
            build_formula_split(output_root / name, records, FORMULA_KEY_MAP,
                                source, name, args.seed, args.force)

    if "md" in args.datasets:
        source = data_root / "formula_pretrain_md_data.json.xz"
        train, valid = split_records(load_xz_json(source), (0.85, 0.15), args.seed)
        for name, records in (("md_train", train), ("md_valid", valid)):
            build_formula_split(output_root / name, records, MD_KEY_MAP,
                                source, name, args.seed, args.force)

    if "mono" in args.datasets:
        source = data_root / "mono_data.json.xz"
        build_mono(output_root / "mono", load_xz_json(source), source, args.force)


if __name__ == "__main__":
    main()
