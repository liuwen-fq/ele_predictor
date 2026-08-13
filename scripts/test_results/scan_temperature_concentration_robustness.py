"""Run temperature--concentration robustness scans for observed candidates.

Candidates are selected from the existing global-high observed contexts, then
their actual salt/solvent ratios are held fixed while the trained model is
queried on a controlled temperature--concentration grid.  The results are
model predictions, not additional experimental observations.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from formula_design.data import FormulaData, collate_data
from formula_design.mol import Molecule
from formula_design.predictor import MolMix, Mono
from scripts.test_results.analyze_conductivity_fragment_attributions import (
    _condition_context, percentile)


LOG_SCALE = 8.3344
LOG_OFFSET = -4.6002


def physical_temperature(normalized: float) -> float:
    return normalized * 373.15 - 273.15


def model_temperature(celsius: float) -> float:
    return (celsius + 273.15) / 373.15


def make_formula(record: dict, concentration: float, temperature: float) -> FormulaData:
    solvents, salts = record["solvents"], record["salts"]
    return FormulaData(
        [item["name"] for item in solvents],
        [Molecule.from_smiles(item["smiles"]).get_mapped_smiles()
         for item in solvents],
        [item["name"] for item in salts],
        [Molecule.from_smiles(item["smiles"]).get_mapped_smiles()
         for item in salts],
        [float(item["molar_ratio"]) for item in solvents],
        [float(item["molar_ratio"]) for item in salts],
        concentration=concentration,
        temperature=model_temperature(temperature),
    )


def load_model(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    checkpoint = Path(config["ckpt_path"])
    pretrain = config["model"]["pretrain"]
    mono = Mono(**pretrain["model_config"])
    mono.load_ckpt(checkpoint / "pretrain.pt")
    model = MolMix(mono, **config["model"]["model"])
    model.load_ckpt(checkpoint / "optimal.pt")
    model.to("cuda").eval()
    return model


def selected_candidates(records: list[dict], windows: list[dict]) -> list[dict]:
    """Pick a real representative record nearest each window midpoint."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        sample = {"record": record,
                  "temperature_c": float(record["temperature"]),
                  "concentration": float(record["salt_molar_ratio"])}
        salt, solvent, temperature, _ = _condition_context(sample)
        grouped[(salt, solvent, temperature)].append(record)
    # A controlled scan changes temperature explicitly.  Windows with the
    # same salt and solvent ratio but a different observed source temperature
    # are evidence replicates, not independent compositions.  Retain only the
    # strongest observed window for each composition.
    selected_windows = {}
    for window in windows:
        composition_key = (window["salt_context"], window["solvent_context"])
        incumbent = selected_windows.get(composition_key)
        if incumbent is None or float(window["mean_bamboo_prediction_ms_cm"]) > float(
                incumbent["mean_bamboo_prediction_ms_cm"]):
            selected_windows[composition_key] = window
    candidates = []
    for index, window in enumerate(sorted(
            selected_windows.values(), key=lambda value: -float(
                value["mean_bamboo_prediction_ms_cm"])), 1):
        key = (window["salt_context"], window["solvent_context"],
               window["temperature_context"])
        values = grouped.get(key, [])
        if not values:
            continue
        midpoint = (float(window["concentration_low"]) +
                    float(window["concentration_high"])) / 2
        representative = min(values, key=lambda value: abs(
            float(value["salt_molar_ratio"]) - midpoint))
        candidates.append({"candidate_id": f"C{index:02d}",
                           "window": window, "record": representative})
    return candidates


@torch.no_grad()
def scan(model, candidates: list[dict], temperatures: list[float],
         concentrations: list[float], batch_size: int) -> list[dict]:
    jobs = []
    for candidate in candidates:
        for temperature in temperatures:
            for concentration in concentrations:
                jobs.append((candidate, temperature, concentration))
    output = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        graphs = collate_data([
            make_formula(candidate["record"], concentration, temperature)
            for candidate, temperature, concentration in batch]).to("cuda")
        prediction = model(graphs)
        conductivity = torch.exp(prediction["conductivity"].squeeze(-1) *
                                 LOG_SCALE + LOG_OFFSET).detach().cpu().tolist()
        explain = prediction["fragment_diagnostics"]["explanations"]["tasks"][
            "conductivity_explain"]
        residual = explain["residual"].detach().cpu().tolist()
        property_sum = explain["property_sum"].detach().cpu().tolist()
        property_by_name = explain["property_by_name"].detach().cpu().tolist()
        names = list(explain["property_names"])
        for (candidate, temperature, concentration), value, fragment, props, prop_items in zip(
                batch, conductivity, residual, property_sum, property_by_name):
            row = {
                "candidate_id": candidate["candidate_id"],
                "temperature_c": temperature,
                "salt_molar_ratio": concentration,
                "conductivity_pred_ms_cm": float(value),
                "fragment_residual_normalized": float(fragment),
                "property_sum_normalized": float(props),
            }
            row.update({f"{name}_effect_normalized": float(effect)
                        for name, effect in zip(names, prop_items)})
            output.append(row)
    return output


def stable_direction(values: list[float], tolerance: float = 0.002) -> str:
    nonzero = [value for value in values if abs(value) >= tolerance]
    if not nonzero:
        return "接近零"
    positive = sum(value > 0 for value in nonzero)
    if max(positive, len(nonzero) - positive) / len(nonzero) < 0.75:
        return "不稳定"
    return "稳定正向" if positive >= len(nonzero) - positive else "稳定负向"


def contiguous_ranges(values: list[float]) -> list[tuple[float, float]]:
    if not values:
        return []
    ranges, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if abs(value - previous - 0.01) > 1e-7:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def summarize(candidates: list[dict], rows: list[dict]) -> list[dict]:
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)
    summary = []
    for candidate in candidates:
        values = by_candidate[candidate["candidate_id"]]
        by_concentration: dict[float, list[dict]] = defaultdict(list)
        for value in values:
            by_concentration[value["salt_molar_ratio"]].append(value)
        concentration_stats = []
        for concentration, points in sorted(by_concentration.items()):
            conductivity = [point["conductivity_pred_ms_cm"] for point in points]
            concentration_stats.append({
                "concentration": concentration,
                "mean": sum(conductivity) / len(conductivity),
                "minimum": min(conductivity),
                "maximum": max(conductivity),
            })
        best_minimum = max(point["minimum"] for point in concentration_stats)
        # A robust plateau keeps at least 90% of the best worst-temperature
        # conductivity; this ranks stability rather than a single sharp peak.
        robust = [point["concentration"] for point in concentration_stats
                  if point["minimum"] >= 0.90 * best_minimum]
        ranges = contiguous_ranges(robust)
        selected = max(ranges, key=lambda item: item[1] - item[0]) if ranges else (None, None)
        selected_rows = [point for point in values if selected[0] is not None and
                         selected[0] - 1e-8 <= point["salt_molar_ratio"] <= selected[1] + 1e-8]
        by_temperature: dict[float, list[float]] = defaultdict(list)
        for point in selected_rows:
            by_temperature[point["temperature_c"]].append(point["conductivity_pred_ms_cm"])
        temperature_means = {temperature: sum(items) / len(items)
                             for temperature, items in by_temperature.items()}
        window = candidate["window"]
        record = candidate["record"]
        row = {
            "candidate_id": candidate["candidate_id"],
            "salt_context": window["salt_context"],
            "solvent_context": window["solvent_context"],
            "observed_temperature_context": window["temperature_context"],
            "representative_source_concentration": record["salt_molar_ratio"],
            "recommended_c_low": selected[0],
            "recommended_c_high": selected[1],
            "robust_window_width": (selected[1] - selected[0]) if selected[0] is not None else 0.0,
            "mean_pred_ms_cm": sum(point["conductivity_pred_ms_cm"] for point in selected_rows) / max(len(selected_rows), 1),
            "worst_case_pred_ms_cm": min(point["conductivity_pred_ms_cm"] for point in selected_rows),
            "best_worst_temperature_pred_ms_cm": best_minimum,
            "temperature_span_ms_cm": max(temperature_means.values()) - min(temperature_means.values()),
            "dc_direction": stable_direction([point["dc_effect_normalized"] for point in selected_rows]),
            "vis_direction": stable_direction([point["vis_effect_normalized"] for point in selected_rows]),
            "Tm_direction": stable_direction([point["Tm_effect_normalized"] for point in selected_rows]),
            "fragment_direction": stable_direction([point["fragment_residual_normalized"] for point in selected_rows]),
            "selected_point_count": len(selected_rows),
        }
        summary.append(row)
    return sorted(summary, key=lambda row: (-row["worst_case_pred_ms_cm"],
                                             -row["mean_pred_ms_cm"]))


def svg_text(x, y, text, size=11, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
            f'font-weight="{weight}">{html.escape(str(text))}</text>')


def write_svg(path: Path, width: int, height: int, elements: list[str]):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                    f'height="{height}" viewBox="0 0 {width} {height}">\n'
                    '<rect width="100%" height="100%" fill="white"/>\n' +
                    "\n".join(elements) + "\n</svg>\n")


def plot_response(rows: list[dict], summary: list[dict], output: Path):
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)
    summary_by_id = {row["candidate_id"]: row for row in summary}
    columns, panel_w, panel_h = 3, 560, 300
    height = math.ceil(len(summary) / columns) * panel_h + 65
    elements = [svg_text(columns * panel_w / 2, 22,
                         "25–60 °C controlled temperature–concentration model scan",
                         16, "middle", "bold"),
                svg_text(15, 43,
                         "colours: temperature; shaded band: robust concentration window; all curves are model predictions",
                         10)]
    palette = {25: "#2166ac", 30: "#4393c3", 35: "#92c5de", 40: "#f4a582",
               45: "#d6604d", 50: "#b2182b", 55: "#762a83", 60: "#4d004b"}
    for index, item in enumerate(summary):
        x0, y0 = 12 + index % columns * panel_w, 58 + index // columns * panel_h
        left, right, top, bottom = x0 + 52, x0 + panel_w - 18, y0 + 58, y0 + panel_h - 38
        points = by_candidate[item["candidate_id"]]
        y_values = [point["conductivity_pred_ms_cm"] for point in points]
        y_min, y_max = max(0.0, min(y_values) - 0.5), max(y_values) + 0.5
        y_scale = (bottom - top) / max(y_max - y_min, 1e-9)
        x_scale = (right - left) / 0.28
        px = lambda value: left + (value - 0.02) * x_scale
        py = lambda value: bottom - (value - y_min) * y_scale
        elements += [f'<rect x="{x0}" y="{y0}" width="{panel_w - 8}" height="{panel_h - 8}" fill="white" stroke="#bbbbbb"/>',
                     svg_text(x0 + (panel_w - 8) / 2, y0 + 17,
                              f'{item["candidate_id"]}: {item["salt_context"]} | {item["solvent_context"]}', 10, "middle", "bold"),
                     svg_text(x0 + (panel_w - 8) / 2, y0 + 32,
                              f'worst={item["worst_case_pred_ms_cm"]:.2f}; c={item["recommended_c_low"]:.2f}–{item["recommended_c_high"]:.2f}', 9, "middle")]
        elements.append(f'<rect x="{px(item["recommended_c_low"]):.1f}" y="{top:.1f}" width="{px(item["recommended_c_high"]) - px(item["recommended_c_low"]):.1f}" height="{bottom-top:.1f}" fill="#f6caca" fill-opacity="0.5"/>')
        for temp in range(25, 61, 5):
            curve = sorted((point for point in points if point["temperature_c"] == temp), key=lambda point: point["salt_molar_ratio"])
            coords = " ".join(f'{px(point["salt_molar_ratio"]):.1f},{py(point["conductivity_pred_ms_cm"]):.1f}' for point in curve)
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{palette[temp]}" stroke-width="1.5"/>')
        elements += [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
                     f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
                     svg_text(left, bottom + 15, "0.02", 8), svg_text(right, bottom + 15, "0.30", 8, "end"),
                     svg_text(left - 4, bottom + 3, f'{y_min:.1f}', 8, "end"), svg_text(left - 4, top + 3, f'{y_max:.1f}', 8, "end")]
    write_svg(output, columns * panel_w + 20, height, elements)


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *["| " + " | ".join(row) + " |" for row in rows]]


def write_report(path: Path, summary: list[dict], rows: list[dict]):
    table = [[item["candidate_id"], item["salt_context"], item["solvent_context"],
              f'{item["recommended_c_low"]:.2f}–{item["recommended_c_high"]:.2f}',
              f'{item["mean_pred_ms_cm"]:.2f}', f'{item["worst_case_pred_ms_cm"]:.2f}',
              f'{item["temperature_span_ms_cm"]:.2f}',
              "/".join([item["dc_direction"], item["vis_direction"], item["Tm_direction"], item["fragment_direction"]])]
             for item in summary]
    lines = ["# 25–60 °C 温度—浓度鲁棒性筛选", "",
             "## 使用方式与边界", "",
             "- 候选来自原报告的全局高预测观测上下文；盐和溶剂内部比例固定为该上下文中的实际代表配方。",
             "- 对每个候选在 `25–60 °C`（5 °C 步长）和 `salt_molar_ratio=0.02–0.30`（0.01 步长）上重新运行主模型。结果是受控**模型扫描**，不是新增实验结果。",
             "- 推荐区间以“跨全部温度的最低预测值”排序：保留达到最佳最低预测值 90% 以上的最长连续浓度段。因此它优先温区稳健性，而不是单一温度尖峰。",
             "- `dc / vis / Tm / fragment` 为解释头在推荐窗口内的方向稳定性；它们是条件关联证据，不是单因素因果证明。",
             "", "## 推荐排序", "",
             *markdown_table(["候选", "盐", "溶剂比例", "推荐 c", "温区平均预测", "温区最差预测", "温度跨度", "dc / vis / Tm / 片段方向"], table),
             "", "## 图：温度—浓度响应", "",
             "每个面板固定一个候选组成；横轴为盐摩尔分数，纵轴为 Bamboo 主预测电导率（mS/cm）。不同颜色对应温度；浅红色为推荐鲁棒浓度区间。", "",
             "![温度—浓度鲁棒性扫描](temperature_concentration_robustness_scan.svg)", ""]
    path.write_text("\n".join(lines))


def update_main_report(path: Path, summary: list[dict]):
    """Append a reproducible robustness-screening section to the main report."""
    text = path.read_text()
    marker = "## 温度—浓度鲁棒性筛选（25–60 °C 模型扫描）"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    table = [[item["candidate_id"], item["salt_context"], item["solvent_context"],
              f'{item["recommended_c_low"]:.2f}–{item["recommended_c_high"]:.2f}',
              f'{item["mean_pred_ms_cm"]:.2f}', f'{item["worst_case_pred_ms_cm"]:.2f}',
              f'{item["temperature_span_ms_cm"]:.2f}',
              "/".join([item["dc_direction"], item["vis_direction"],
                        item["Tm_direction"], item["fragment_direction"]])]
             for item in summary]
    addition = ["", marker, "",
                "本节固定候选的盐/溶剂内部比例，在 `25–60 °C`（5 °C 步长）及 `salt_molar_ratio=0.02–0.30`（0.01 步长）上重新运行 Bamboo 主模型。它是模型受控扫描，不是新增实验数据。重复的观测温度窗口已按相同盐种与溶剂比例合并，因此每行代表一个独立组成。推荐区间为跨温度最低预测达到该配方最佳最低预测 90% 以上的最长连续区间。", "",
                *markdown_table(["候选", "盐", "溶剂比例", "推荐 c", "温区平均预测", "温区最差预测", "温度跨度", "dc / vis / Tm / 片段方向"], table),
                "", "**解释性读取。** `dc / vis / Tm / 片段方向` 分别是在推荐温度—浓度窗口内的解释头方向稳定性。优先候选应同时具有较高的温区最差预测、较宽的浓度窗口和稳定的解释方向；其中 `vis` 不稳定的体系（例如 LiTFSI/PC、LiBOB/PC）不宜作为首选稳健候选。LiTFSI/AN 在本扫描中具有最高温区平均与最差预测，且四类解释项均稳定正向；LiTFSI/Glyme 的 `dc` 为稳定负向，说明其高预测更依赖迁移率/片段残差等其他条件贡献，不能简单归因于介电常数。", "",
                "![25–60 °C 温度—浓度鲁棒性扫描](temperature_concentration_robustness/temperature_concentration_robustness_scan.svg)", "",
                "完整逐网格预测、候选排序及单独方法报告见 `temperature_concentration_robustness/`。", ""]
    path.write_text(text + "\n".join(addition))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/test_results/property_fragment_predict.yaml")
    parser.add_argument("--source", default="run/property_fragment_explanation/results/ft_test_predictions.json")
    parser.add_argument("--windows", default="run/property_fragment_attribution_v2/recommended_concentration_windows.csv")
    parser.add_argument("--output-dir", default="run/property_fragment_attribution_v2/temperature_concentration_robustness")
    parser.add_argument("--main-report", default="run/property_fragment_attribution_v2/fragment_attribution_report_zh.md")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    with open(args.windows, newline="") as handle:
        windows = list(csv.DictReader(handle))
    candidates = selected_candidates(source, windows)
    temperatures = list(range(25, 61, 5))
    concentrations = [round(0.02 + index * 0.01, 2) for index in range(29)]
    model = load_model(Path(args.config))
    rows = scan(model, candidates, temperatures, concentrations, args.batch_size)
    summary = summarize(candidates, rows)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "scan_predictions.json").write_text(json.dumps(rows, indent=2))
    with open(output / "robustness_ranking.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    plot_response(rows, summary, output / "temperature_concentration_robustness_scan.svg")
    write_report(output / "temperature_concentration_robustness_report_zh.md", summary, rows)
    update_main_report(Path(args.main_report), summary)
    print(f"Wrote {len(rows)} model scan points for {len(candidates)} candidates to {output}")


if __name__ == "__main__":
    main()
