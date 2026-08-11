"""Summarize condition-aware solvent-property contributions.

The values are normalized-log-conductivity terms from the independent
explanation head.  They are useful for interpreting model behavior, not as
standalone causal property effects.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from explain_experimental_fragments import conductivity_ms_cm, percentile


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize property terms from property-fragment explanations")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.quantile < 0.5:
        raise ValueError("--quantile must be in (0, 0.5)")

    records = json.loads(Path(args.input).read_text())
    selected = [record for record in records
                if record.get("conductivity_mask", False)
                and record.get("property_explanations")]
    if not selected:
        raise ValueError("no property_explanations with experimental labels found")
    values = [conductivity_ms_cm(float(record["conductivity"]))
              for record in selected]
    low_cutoff = percentile(values, args.quantile)
    high_cutoff = percentile(values, 1 - args.quantile)
    grouped = defaultdict(list)
    solvent_rows = []
    for record in selected:
        target = conductivity_ms_cm(float(record["conductivity"]))
        group = "high" if target >= high_cutoff else "low" if target <= low_cutoff else "middle"
        explanation = record["property_explanations"]
        for name, effect in explanation["property_sum_by_name_normalized"].items():
            grouped[name].append((group, float(effect)))
        for solvent in explanation["solvent_property_contributions"]:
            for name, effect in solvent["property_effects_normalized"].items():
                solvent_rows.append({
                    "conductivity_group": group,
                    "molecule_name": solvent["molecule_name"],
                    "molar_ratio": solvent["molar_ratio"],
                    "property": name,
                    "property_value_normalized": solvent[
                        "property_values_normalized"][name],
                    "property_effect_normalized": effect,
                })
    summary = []
    for name, rows in sorted(grouped.items()):
        high = [value for group, value in rows if group == "high"]
        low = [value for group, value in rows if group == "low"]
        effects = [value for _, value in rows]
        summary.append({
            "property": name,
            "formula_count": len(effects),
            "mean_effect_normalized": mean(effects),
            "positive_effect_fraction": sum(value > 0 for value in effects) /
            len(effects),
            "high_group_mean_effect_normalized": mean(high) if high else "",
            "low_group_mean_effect_normalized": mean(low) if low else "",
            "high_minus_low_effect_normalized": (
                mean(high) - mean(low)) if high and low else "",
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "property_contribution_summary.csv", summary)
    write_csv(output_dir / "solvent_property_contributions.csv", solvent_rows)
    (output_dir / "property_contribution_report.md").write_text(
        "# 条件感知性质贡献汇总\n\n"
        f"- 有实验标签样本数：{len(selected)}\n"
        f"- 低/高电导率阈值：{low_cutoff:.4g} / {high_cutoff:.4g} mS/cm\n\n"
        "性质贡献来自冻结分子多任务模型的归一化性质输出，经解释头按盐种、浓度、"
        "温度和溶剂比例调制后得到。其正负号是模型条件关联，不能单独视为化学因果。\n")
    print(f"Wrote property contribution analysis to: {output_dir}")


if __name__ == "__main__":
    main()
