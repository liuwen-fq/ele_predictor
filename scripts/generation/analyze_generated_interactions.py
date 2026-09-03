#!/usr/bin/env python3
"""Uniform-oracle interaction attribution for generated G0/G2-b formulas.

The released G0 JSON is originally scored by the released Bamboo predictor,
whereas G2-b is scored by the current dual-task oracle.  This script rebuilds
both formula sets from their decoded BOW vectors and evaluates them with the
*same current explanation oracle*.  It then bins formulas by predicted sigma
and reports whether target-bin interaction patterns are stable under bootstrap
resampling.  It is an attribution/diagnostic analysis, not a causal test.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from formula_design.data import FormulaData, collate_data
from formula_design.fragment import attach_fragment_attention
from formula_design.mol import Molecule
from formula_design.predictor import MolMix, Mono


SIGMA_EDGES = np.asarray([0., 5., 10., 15., 20., 25., 30., 40., np.inf])
SIGMA_LABELS = ("0–5", "5–10", "10–15", "15–20", "20–25", "25–30", "30–40", "≥40")
TASKS = ("conductivity_explain", "anion_ratio_explain")
COMPONENT_KEYS = ("main_effect_sum_normalized", "interaction_sum_normalized",
                  "property_sum_normalized")


def load_oracle(config_path: Path, checkpoint: Path, device: str):
    config = yaml.safe_load(config_path.read_text())
    mono = Mono(**config["model"]["pretrain"]["model_config"])
    mono.load_ckpt(str(checkpoint / "pretrain.pt"))
    model = MolMix(mono, **config["model"]["model"])
    model.load_ckpt(str(checkpoint / "optimal.pt"))
    return model.eval().to(device)


def formula_from_bow(bow, solvents, salts, temperature_c, concentration):
    bow = np.asarray(bow, dtype=float)
    n_solvents = len(solvents)
    solvent_ids = np.flatnonzero(bow[:n_solvents] > 0).tolist()
    salt_ids = np.flatnonzero(bow[n_solvents:] > 0).tolist()
    if not solvent_ids or not salt_ids:
        raise ValueError("decoded BOW has no active solvent or salt")
    return FormulaData(
        [solvents[index]["name"] for index in solvent_ids],
        [Molecule.from_smiles(solvents[index]["smiles"]).get_mapped_smiles()
         for index in solvent_ids],
        [salts[index]["name"] for index in salt_ids],
        [Molecule.from_smiles(salts[index]["smiles"]).get_mapped_smiles()
         for index in salt_ids],
        [float(bow[index]) for index in solvent_ids],
        [float(bow[n_solvents + index]) for index in salt_ids],
        temperature=(float(temperature_c) + 273.15) / 373.15,
        concentration=float(concentration),
    )


def annotate_fragments(graph, diagnostics):
    solvent_diagnostics = diagnostics.get("solvents")
    if solvent_diagnostics is None:
        raise RuntimeError("the current oracle did not return solvent fragment diagnostics")
    if "fragment_metadata" not in graph.solv_graphs:
        raise RuntimeError("formula graph has no fragment metadata")
    return attach_fragment_attention(
        graph.solv_graphs.fragment_metadata,
        solvent_diagnostics["attention"].detach().cpu().tolist(),
        solvent_diagnostics["neutral_only"],
    )


def fragment_lookup(annotated_formulas):
    lookup, offset = {}, 0
    for formula_index, formula in enumerate(annotated_formulas):
        for molecule_index, molecule in enumerate(formula):
            for fragment in molecule["fragments"]:
                lookup[offset] = {
                    "formula_index": formula_index,
                    "molecule_index": molecule_index,
                    "molecule_name": molecule.get("molecule_name", "unknown"),
                    "fragment_smiles": fragment.get("smiles", ""),
                }
                offset += 1
    return lookup


def serialize_task(task_data, task_name, annotated_formulas):
    """Keep formula-level terms plus every cross-molecule pair diagnostic."""
    formula_count = len(annotated_formulas)
    def cpu_list(name, default=0.):
        value = task_data.get(name)
        if value is None:
            return [default] * formula_count
        return value.detach().cpu().reshape(-1).tolist()

    output = []
    for index, values in enumerate(zip(
            cpu_list("main_sum"), cpu_list("interaction_sum"),
            cpu_list("property_sum"), cpu_list("residual"),
            cpu_list("explanation_prediction"))):
        main, interaction, property_sum, residual, prediction = values
        output.append({
            "task": task_name,
            "main_effect_sum_normalized": float(main),
            "interaction_sum_normalized": float(interaction),
            "property_sum_normalized": float(property_sum),
            "total_residual_normalized": float(residual),
            "explanation_prediction_normalized": float(prediction),
            "top_fragment_pairs": [],
        })
    pair_indices = task_data.get("pair_indices")
    if pair_indices is None:
        return output
    lookup = fragment_lookup(annotated_formulas)
    pair_indices = pair_indices.detach().cpu().tolist()
    pair_formula_indices = task_data["pair_formula_index"].detach().cpu().tolist()
    pair_effects = task_data["pair_effect"].detach().cpu().tolist()
    for (left, right), formula_index, effect in zip(
            pair_indices, pair_formula_indices, pair_effects):
        left_info, right_info = lookup[int(left)], lookup[int(right)]
        if left_info["formula_index"] != formula_index or right_info["formula_index"] != formula_index:
            raise RuntimeError("fragment pair does not map to its formula")
        output[formula_index]["top_fragment_pairs"].append({
            "effect_normalized": float(effect),
            "left": {key: left_info[key] for key in ("molecule_index", "molecule_name", "fragment_smiles")},
            "right": {key: right_info[key] for key in ("molecule_index", "molecule_name", "fragment_smiles")},
        })
    return output


def load_group_records(root: Path, group: str):
    source = root / group / "generated" / "generated_formulas.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    records = json.loads(source.read_text())
    pool = root / group / "candidate_pool"
    solvents = json.loads((pool / "solvents.json").read_text())
    salts = json.loads((pool / "salts.json").read_text())
    return records, solvents, salts, source


def score_group(model, records, solvents, salts, group, batch_size, device):
    rows = []
    for start in range(0, len(records), batch_size):
        batch_source = records[start:start + batch_size]
        formulas = [formula_from_bow(
            item["bow"], solvents, salts,
            item.get("temperature_c", 25.), item.get("salt_molar_ratio", .1))
            for item in batch_source]
        graph = collate_data(formulas).to(device)
        with torch.no_grad():
            prediction = model(graph)
        log_sigma = prediction["conductivity"].detach().cpu().reshape(-1).numpy()
        sigma = np.exp(log_sigma * 8.3344 - 4.6002)
        ratio = prediction["anion_ratio"].detach().cpu().reshape(-1).numpy()
        diagnostics = prediction["fragment_diagnostics"]
        annotated = annotate_fragments(graph, diagnostics)
        explanation = diagnostics.get("explanations")
        if explanation is None:
            raise RuntimeError("current oracle has no property/fragment explanation output")
        tasks = explanation["tasks"]
        serialized = {
            task: serialize_task(tasks[task], task, annotated)
            for task in TASKS
        }
        for offset, source in enumerate(batch_source):
            rows.append({
                "group": group,
                "source_index": start + offset,
                "temperature_c": float(source.get("temperature_c", 25.)),
                "salt_molar_ratio": float(source.get("salt_molar_ratio", .1)),
                "target_sigma_ms_cm": float(source.get("target_conductivity_ms_cm", np.nan)),
                "target_anion_ratio": float(source.get("anion_ratio_condition", np.nan)),
                "sigma_pred_ms_cm_current_oracle": float(sigma[offset]),
                "anion_ratio_pred_current_oracle": float(ratio[offset]),
                "solvents": [{"name": item["name"], "smiles": item["smiles"],
                              "molar_ratio": float(source["bow"][index])}
                             for index, item in enumerate(solvents)
                             if float(source["bow"][index]) > .02],
                "salts": [{"name": item["name"], "smiles": item["smiles"],
                           "molar_ratio": float(source["bow"][len(solvents) + index])}
                          for index, item in enumerate(salts)
                          if float(source["bow"][len(solvents) + index]) > .02],
                "conductivity_fragment_explanation": serialized["conductivity_explain"][offset],
                "anion_ratio_fragment_explanation": serialized["anion_ratio_explain"][offset],
            })
        print(f"{group}: current-oracle explanation {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return rows


def mean_ci(values, rng, bootstrap):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    if len(values) == 1 or bootstrap <= 0:
        return mean, mean, mean
    sample_indices = rng.integers(0, len(values), size=(bootstrap, len(values)))
    samples = values[sample_indices].mean(axis=1)
    return mean, float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def pair_key(pair):
    left, right = pair["left"], pair["right"]
    values = sorted([
        (left["molecule_name"], left["fragment_smiles"]),
        (right["molecule_name"], right["fragment_smiles"]),
    ])
    return " || ".join(f"{name}::{smiles}" for name, smiles in values)


def bin_index(sigma):
    index = int(np.digitize(float(sigma), SIGMA_EDGES) - 1)
    return min(max(index, 0), len(SIGMA_LABELS) - 1)


def summarize(rows, bootstrap, seed, min_pair_formulas, sign_threshold):
    rng = np.random.default_rng(seed)
    by_group_bin = defaultdict(list)
    for row in rows:
        if np.isfinite(row["sigma_pred_ms_cm_current_oracle"]):
            by_group_bin[(row["group"], bin_index(row["sigma_pred_ms_cm_current_oracle"]))].append(row)
    summary_rows, pair_rows = [], []
    groups = sorted({row["group"] for row in rows})
    for group in groups:
        for current_bin, label in enumerate(SIGMA_LABELS):
            entries = by_group_bin[(group, current_bin)]
            base = {"group": group, "sigma_bin": label, "n_formulas": len(entries)}
            for task, prefix in (("conductivity_fragment_explanation", "sigma"),
                                 ("anion_ratio_fragment_explanation", "anion_ratio")):
                task_rows = [entry[task] for entry in entries]
                for key in COMPONENT_KEYS:
                    mean, lo, hi = mean_ci([entry[key] for entry in task_rows], rng, bootstrap)
                    base[f"{prefix}_{key}_mean"] = mean
                    base[f"{prefix}_{key}_ci_low"] = lo
                    base[f"{prefix}_{key}_ci_high"] = hi
            summary_rows.append(base)

            for task, target in (("conductivity_fragment_explanation", "sigma"),
                                 ("anion_ratio_fragment_explanation", "anion_ratio")):
                effects = defaultdict(list)
                formula_indices = defaultdict(set)
                for entry in entries:
                    for pair in entry[task]["top_fragment_pairs"]:
                        key = pair_key(pair)
                        effects[key].append(float(pair["effect_normalized"]))
                        formula_indices[key].add(entry["source_index"])
                for key, values in effects.items():
                    n_formulas = len(formula_indices[key])
                    mean, lo, hi = mean_ci(values, rng, bootstrap)
                    positive = float(np.mean(np.asarray(values) > 0))
                    sign_consistency = max(positive, 1. - positive)
                    stable = (n_formulas >= min_pair_formulas
                              and sign_consistency >= sign_threshold
                              and ((lo > 0) or (hi < 0)))
                    pair_rows.append({
                        "group": group, "sigma_bin": label, "task": target,
                        "pair": key, "n_pair_records": len(values),
                        "n_formulas": n_formulas, "mean_effect_normalized": mean,
                        "ci_low": lo, "ci_high": hi,
                        "positive_fraction": positive,
                        "sign_consistency": sign_consistency,
                        "stable": stable,
                    })
    return summary_rows, pair_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_components(summary_rows, output):
    lookup = {(row["group"], row["sigma_bin"]): row for row in summary_rows}
    groups = [group for group in ("G0", "G2-b")
              if any(row["group"] == group for row in summary_rows)]
    fig, axes = plt.subplots(2, len(groups), figsize=(6.5 * len(groups), 8.2),
                             sharex=True, squeeze=False)
    labels = ("main", "interaction", "property")
    colors = ("#2563eb", "#d97706", "#059669")
    x = np.arange(len(SIGMA_LABELS))
    width = .24
    for col, group in enumerate(groups):
        for row_index, (prefix, title, y_label) in enumerate((
                ("sigma", "σ explanation terms", "normalized σ contribution"),
                ("anion_ratio", "Anion-ratio explanation terms", "normalized ratio-logit contribution"))):
            ax = axes[row_index, col]
            for offset, (component, label, color) in enumerate(zip(COMPONENT_KEYS, labels, colors)):
                values = [lookup.get((group, bin_label), {}).get(
                    f"{prefix}_{component}_mean", np.nan) for bin_label in SIGMA_LABELS]
                errors = []
                for bin_label, value in zip(SIGMA_LABELS, values):
                    item = lookup.get((group, bin_label), {})
                    lo, hi = item.get(f"{prefix}_{component}_ci_low", np.nan), item.get(f"{prefix}_{component}_ci_high", np.nan)
                    errors.append([max(0., value - lo), max(0., hi - value)] if np.isfinite(value) and np.isfinite(lo) and np.isfinite(hi) else [0., 0.])
                ax.bar(x + (offset - 1) * width, values, width, label=label,
                       color=color, alpha=.84,
                       yerr=np.asarray(errors).T if errors else None, capsize=2)
            ax.axhline(0, color="#374151", lw=.8)
            ax.set_title(f"{group}: {title}")
            ax.set_ylabel(y_label)
            ax.grid(axis="y", alpha=.2)
            if row_index == 0 and col == 0:
                ax.legend(frameon=False, ncol=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xticks(x, SIGMA_LABELS, rotation=25)
        ax.set_xlabel("Current-oracle predicted σ bin (mS/cm)")
    fig.suptitle("Uniform-current-oracle explanation terms by generated σ bin", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(output, condition_root, source_files, summary_rows, pair_rows,
                 bootstrap, min_pair_formulas, sign_threshold):
    del condition_root, source_files
    # Target conditions are stated by the caller because summary rows are aggregates.
    stable_rows = [row for row in pair_rows if row["stable"]]
    counts = defaultdict(int)
    for row in stable_rows:
        counts[(row["group"], row["sigma_bin"], row["task"])] += 1
    focus = ("15–20", "20–25", "25–30")
    lines = [
        "# G0 与 G2-b：统一 oracle 片段交互归因与 σ 分箱分析", "",
        "## 分析目的", "",
        "本分析以当前双任务解释 oracle 对 G0 与 G2-b 的已生成配方重新评分并分解贡献。"
        "它消除了原始结果中 G0 使用发布 Bamboo predictor、G2-b 使用当前 oracle 的评分器差异。"
        "结果反映当前 oracle 内的稳定解释模式，不构成实验因果证明。", "",
        "## 方法", "",
        f"- 每组全部生成配方均按当前 oracle 预测 σ 分入固定区间：{', '.join(SIGMA_LABELS)} mS/cm。",
        "- 对每个 σ bin 分别汇总 σ 与 anion ratio 解释头的主片段项、跨溶剂片段对交互项和物性项。",
        f"- 均值的 95% CI 由 {bootstrap} 次配方级 bootstrap 得到。",
        f"- 片段对被标为稳定，需同时满足：至少出现于 {min_pair_formulas} 个配方、符号一致性 ≥ {sign_threshold:.0%}，且 bootstrap 95% CI 不跨零。",
        "- 交互项均为解释头的归一化贡献；σ 项位于模型的对数 σ 残差尺度，anion ratio 项位于 logit 尺度，不能直接解释为 mS/cm 或 ratio 的因果改变量。", "",
        "## 分箱贡献图", "",
        "![σ-bin explanation terms](sigma_bin_explanation_terms.png)", "",
        "## 稳定交互模式计数", "",
        "| group | σ bin | stable σ pairs | stable anion-ratio pairs |",
        "| --- | --- | ---: | ---: |",
    ]
    for group in ("G0", "G2-b"):
        for sigma_bin in focus:
            lines.append("| " + " | ".join([group, sigma_bin,
                str(counts[(group, sigma_bin, "sigma")]),
                str(counts[(group, sigma_bin, "anion_ratio")])]) + " |")
    lines.extend([
        "", "稳定片段对的完整清单及其出现配方数、符号一致性和 bootstrap CI 位于"
        "[CSV](stable_fragment_pair_interactions.csv)。所有配方的统一 oracle 归因位于"
        "`*_current_oracle_explanations.json`。", "",
        "## 如何解释 G2-b 的 20–25 mS/cm 富集", "",
        "只有当 G2-b 的 20–25 bin 同时满足以下条件，才能把交互称为其目标富集的模型内支持证据："
        "(1) σ/r 交互均值的 CI 不跨零；(2) 稳定片段对数量或效应模式区别于相邻 15–20、25–30 bin；"
        "(3) 这些模式在统一 oracle 下相较 G0 可重复。否则，G2-b 的集中更应解释为 synthetic 标签分布与重训生成器的条件分布对齐，而不能归因于片段交互。",
    ])
    (output / "interaction_attribution_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--groups", nargs="+", default=["G0", "G2-b"])
    parser.add_argument("--oracle-config", default="config/test_results/anion_ratio_property_fragment_predict.yaml")
    parser.add_argument("--oracle-checkpoint", default="run/anion_ratio_property_fragment_explanation/ckpt_formula_ft")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-pair-formulas", type=int, default=20)
    parser.add_argument("--sign-threshold", type=float, default=.75)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    output = Path(args.output_dir) if args.output_dir else root / "interaction_attribution_current_oracle"
    output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this explanation analysis in the bamboo GPU environment")
    device = args.device
    model = load_oracle(Path(args.oracle_config), Path(args.oracle_checkpoint), device)
    all_rows, source_files = [], {}
    for group in args.groups:
        records, solvents, salts, source = load_group_records(root, group)
        scored = score_group(model, records, solvents, salts, group, args.batch_size, device)
        path = output / f"{group}_current_oracle_explanations.json"
        path.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
        all_rows.extend(scored)
        source_files[group] = str(source)
    summary_rows, pair_rows = summarize(
        all_rows, args.bootstrap, args.seed, args.min_pair_formulas, args.sign_threshold)
    write_csv(output / "sigma_bin_explanation_summary.csv", summary_rows)
    stable_rows = [row for row in pair_rows if row["stable"]]
    write_csv(output / "stable_fragment_pair_interactions.csv", stable_rows)
    write_csv(output / "all_fragment_pair_interactions.csv", pair_rows)
    plot_components(summary_rows, output / "sigma_bin_explanation_terms.png")
    write_report(output, root, source_files, summary_rows, pair_rows,
                 args.bootstrap, args.min_pair_formulas, args.sign_threshold)
    provenance = {
        "experiment_root": str(root), "groups": args.groups,
        "source_generated_formula_json": source_files,
        "oracle_config": str(args.oracle_config),
        "oracle_checkpoint": str(args.oracle_checkpoint),
        "scoring": "both groups rescored with the current dual-task explanation oracle",
        "sigma_edges_ms_cm": SIGMA_EDGES.tolist(), "bootstrap": args.bootstrap,
        "seed": args.seed, "min_pair_formulas": args.min_pair_formulas,
        "sign_threshold": args.sign_threshold,
    }
    (output / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote uniform-oracle interaction attribution analysis to {output}")


if __name__ == "__main__":
    main()
