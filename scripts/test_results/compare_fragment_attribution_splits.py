#!/usr/bin/env python3
"""Compare high/low chemical-motif attribution patterns across data splits.

This deliberately aggregates motifs rather than ranking individual formulas, so
frequent solvents (for example AN) cannot dominate the conclusion simply by
appearing many times in one split.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("train", "valid", "test")


def portable_formula_label(value):
    """Convert the few Chinese annotations in group formulae for SVG labels."""
    return (value.replace("（环）", " (cyclic)")
            .replace("；", ";")
            .replace("（", "(").replace("）", ")"))


def read_rows(directory: Path):
    with (directory / "chemical_motif_attribution_summary.csv").open(
            newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row.get(key, 0.0) or 0.0)


def write_csv(path: Path, rows):
    fields = [
        "motif_key", "fragment_chinese_name", "fragment_group_formula",
        "solvent_molecules", "present_splits", "direction_consistency",
        "mean_high_minus_low_presence", "train_high_minus_low_presence",
        "valid_high_minus_low_presence", "test_high_minus_low_presence",
        "mean_direct_effect", "train_direct_effect", "valid_direct_effect",
        "test_direct_effect",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path: Path):
    rows = sorted(rows, key=lambda row: abs(number(
        row, "mean_high_minus_low_presence")), reverse=True)[:16]
    if not rows:
        return
    # Keep the SVG portable: the server's Matplotlib installation has no CJK
    # font.  Chemical group formulae are unambiguous and render everywhere;
    # the adjacent Markdown/CSV retain the Chinese names.
    labels = [portable_formula_label(row["fragment_group_formula"])
              for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.48 * len(rows))))
    offsets = (-0.24, 0.0, 0.24)
    colors = ("#4477AA", "#66AA55", "#CC6677")
    for split, offset, color in zip(SPLITS, offsets, colors):
        values = [number(row, f"{split}_high_minus_low_presence")
                  for row in rows]
        ax.barh(y + offset, values, height=0.22, label=split,
                color=color, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("High-group presence − low-group presence (within split)")
    ax.set_title("Motif enrichment consistency: train / valid / test")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--master-report", type=Path)
    args = parser.parse_args()

    by_split = {
        "train": read_rows(args.train_dir),
        "valid": read_rows(args.valid_dir),
        "test": read_rows(args.test_dir),
    }
    merged = {}
    for split, rows in by_split.items():
        for row in rows:
            merged.setdefault(row["motif_key"], {})[split] = row

    rows = []
    for motif_key, split_rows in merged.items():
        exemplar = next(iter(split_rows.values()))
        deltas = {
            split: number(row, "high_presence_rate") -
            number(row, "low_presence_rate")
            for split, row in split_rows.items()
        }
        direct = {
            split: number(row, "mean_main_effect_normalized")
            for split, row in split_rows.items()
        }
        nonzero_signs = {int(np.sign(value)) for value in deltas.values()
                         if abs(value) >= 0.02}
        if len(split_rows) == 3 and len(nonzero_signs) == 1:
            consistency = "三集合方向一致"
        elif len(nonzero_signs) == 1 and len(split_rows) >= 2:
            consistency = "至少两集合方向一致"
        else:
            consistency = "跨集合不稳定/差异小"
        row = {
            "motif_key": motif_key,
            "fragment_chinese_name": exemplar["fragment_chinese_name"],
            "fragment_group_formula": exemplar["fragment_group_formula"],
            "solvent_molecules": exemplar["solvent_molecules"],
            "present_splits": "/".join(split_rows),
            "direction_consistency": consistency,
            "mean_high_minus_low_presence": np.mean(list(deltas.values())),
            "mean_direct_effect": np.mean(list(direct.values())),
        }
        for split in SPLITS:
            row[f"{split}_high_minus_low_presence"] = deltas.get(split, 0.0)
            row[f"{split}_direct_effect"] = direct.get(split, 0.0)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["direction_consistency"] != "三集合方向一致",
                               -abs(number(row, "mean_high_minus_low_presence"))))
    write_csv(args.output_dir / "cross_split_motif_stability.csv", rows)
    plot(rows, args.output_dir / "cross_split_motif_stability.svg")

    stable = [row for row in rows
              if row["direction_consistency"] == "三集合方向一致"]
    high = [row for row in stable
            if number(row, "mean_high_minus_low_presence") > 0]
    low = [row for row in stable
           if number(row, "mean_high_minus_low_presence") < 0]
    lines = [
        "# Train/Valid/Test 化学基元归因一致性",
        "",
        "三个数据集均使用各自带实验电导率标签的配方，并在各自内部按高/低四分位组比较；不按单个配方排名，因此不会因 AN 等重复溶剂而得出偏置结论。",
        "",
        "- [训练集完整报告](../train/fragment_attribution_report_zh.md)",
        "- [验证集完整报告](../valid/fragment_attribution_report_zh.md)",
        "- [测试集完整报告](../fragment_attribution_report_zh.md)",
        "- [跨集合富集一致性图](cross_split_motif_stability.svg)",
        "",
        "## 跨集合稳定的高电导率富集基元",
        "",
    ]
    for row in high[:12]:
        lines.append(
            f"- {row['fragment_chinese_name']}（{row['fragment_group_formula']}）："
            f"平均高−低出现率差 {number(row, 'mean_high_minus_low_presence'):.3f}；"
            f"平均直接贡献 {number(row, 'mean_direct_effect'):.4f}。")
    lines += ["", "## 跨集合稳定的低电导率富集基元", ""]
    for row in low[:12]:
        lines.append(
            f"- {row['fragment_chinese_name']}（{row['fragment_group_formula']}）："
            f"平均高−低出现率差 {number(row, 'mean_high_minus_low_presence'):.3f}；"
            f"平均直接贡献 {number(row, 'mean_direct_effect'):.4f}。")
    lines += [
        "",
        "解释：富集方向表示该基元在高/低电导率配方中更常出现；直接贡献表示在当前模型条件下的局部增减方向。两者一致才可作为较强候选，仍需结合盐、浓度、温度和分子物性进行实验验证。",
    ]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "cross_split_motif_stability_report_zh.md").write_text(
        report, encoding="utf-8")

    if args.master_report:
        marker_start = "<!-- cross-split-attribution-start -->"
        marker_end = "<!-- cross-split-attribution-end -->"
        master = args.master_report.read_text(encoding="utf-8")
        section = (f"\n{marker_start}\n## 训练/验证/测试集的完整归因与一致性\n\n"
                   "训练与验证集不再只列电导率前十配方，而是与测试集采用相同的高/低组片段、"
                   "片段交互、物性和浓度归因流程。跨集合结论见"
                   "[一致性报告](cross_split/cross_split_motif_stability_report_zh.md)。"
                   "\n\n- [训练集完整归因](train/fragment_attribution_report_zh.md)\n"
                   "- [验证集完整归因](valid/fragment_attribution_report_zh.md)\n"
                   "- [富集一致性图](cross_split/cross_split_motif_stability.svg)\n"
                   f"{marker_end}\n")
        if marker_start in master and marker_end in master:
            start = master.index(marker_start)
            end = master.index(marker_end) + len(marker_end)
            master = master[:start] + section.strip() + master[end:]
        else:
            master += section
        args.master_report.write_text(master, encoding="utf-8")


if __name__ == "__main__":
    main()
