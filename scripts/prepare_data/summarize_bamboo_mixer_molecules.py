#!/usr/bin/env python3
"""Summarize molecules used by Bamboo-Mixer MD and experimental formula data.

The released ``formula_finetune_data`` is a merged convenience file.  This
script deliberately compares the two raw sources that constitute it:

* ``formula_finetune_exp_data``: experimental conductivity fine-tuning data;
* ``formula_pretrain_md_data``: MD formula pre-training data.

It writes one complete CSV, source-specific CSV views, an audit JSON, and a
Chinese Markdown report.  Molecules are identified by canonical input SMILES;
all observed names are retained as aliases rather than assuming a name is a
unique chemical identifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    from rdkit import Chem
except ImportError:  # The inventory itself should remain usable without RDKit.
    Chem = None


SOURCE_FILES = {
    "实验微调子集": "formula_finetune_exp_data.json.xz",
    "MD预训练": "formula_pretrain_md_data.json.xz",
}
MERGED_FILE = "formula_finetune_data.json.xz"


def read_records(root: Path, filename: str) -> list[dict[str, Any]]:
    compressed = root / "data_oss" / filename
    plain = root / "run" / filename.removesuffix(".xz")
    if compressed.is_file():
        with lzma.open(compressed, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    if plain.is_file():
        with plain.open(encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"Neither {compressed} nor {plain} exists")


def collect(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Collect one row per (component role, SMILES)."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for key, label in (("solvents", "溶剂"), ("salts", "盐")):
            seen_in_formula: set[str] = set()
            for component in record.get(key, []):
                name = str(component.get("name", "")).strip()
                smiles = str(component.get("smiles", "")).strip()
                if not smiles:
                    raise ValueError(f"Missing SMILES in {key}: {component!r}")
                identity = (label, smiles)
                entry = result.setdefault(identity, {
                    "类别": label,
                    "SMILES": smiles,
                    "名称": set(),
                    "配方出现次数": 0,
                })
                if name:
                    entry["名称"].add(name)
                # Defensive de-duplication for malformed records that list a
                # component twice in a single formula.
                if smiles not in seen_in_formula:
                    entry["配方出现次数"] += 1
                    seen_in_formula.add(smiles)
    return result


def source_classification(exp_count: int, md_count: int) -> str:
    if exp_count and md_count:
        return "两者均有"
    if exp_count:
        return "仅实验微调子集"
    return "仅MD预训练"


def make_rows(exp: dict[tuple[str, str], dict[str, Any]],
              md: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in sorted(set(exp) | set(md), key=lambda x: (x[0], x[1])):
        exp_entry, md_entry = exp.get(identity), md.get(identity)
        names = sorted((exp_entry or {"名称": set()})["名称"] |
                       (md_entry or {"名称": set()})["名称"])
        exp_count = 0 if exp_entry is None else exp_entry["配方出现次数"]
        md_count = 0 if md_entry is None else md_entry["配方出现次数"]
        rows.append({
            "类别": identity[0],
            "名称": " | ".join(names) or "(未命名)",
            "SMILES": identity[1],
            "实验微调子集_配方出现次数": exp_count,
            "MD预训练_配方出现次数": md_count,
            "来源分类": source_classification(exp_count, md_count),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["类别", "名称", "SMILES", "实验微调子集_配方出现次数",
              "MD预训练_配方出现次数", "来源分类"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_by_role(rows: list[dict[str, Any]], source: str) -> dict[str, int]:
    field = f"{source}_配方出现次数"
    return {
        role: sum(1 for row in rows if row["类别"] == role and row[field])
        for role in ("溶剂", "盐")
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], exp_records: int,
                   md_records: int, merged_records: int | None) -> None:
    exp_roles = count_by_role(rows, "实验微调子集")
    md_roles = count_by_role(rows, "MD预训练")
    shared = sum(row["来源分类"] == "两者均有" for row in rows)
    exp_only = sum(row["来源分类"] == "仅实验微调子集" for row in rows)
    md_only = sum(row["来源分类"] == "仅MD预训练" for row in rows)
    lines = [
        "# Bamboo-Mixer 配方数据分子清单",
        "",
        "## 数据口径",
        "",
        "- **实验微调子集**：`formula_finetune_exp_data`，具有实验电导率监督。",
        "- **MD 预训练**：`formula_pretrain_md_data`，具有 MD 结构/输运相关监督。",
        "- `formula_finetune_data` 是上述两份数据的合并便利文件，不作为独立来源计数。",
        "- 以“组件类别（溶剂/盐）+ SMILES”为一个条目；同一 SMILES 的不同命名保留为别名。",
        "",
        "## 规模与重叠",
        "",
        "| 项目 | 实验微调子集 | MD预训练 |",
        "| --- | ---: | ---: |",
        f"| 配方记录数 | {exp_records:,} | {md_records:,} |",
        f"| 唯一溶剂数 | {exp_roles['溶剂']} | {md_roles['溶剂']} |",
        f"| 唯一盐数 | {exp_roles['盐']} | {md_roles['盐']} |",
        f"| 合并后唯一组件数 | {len(rows)} | - |",
        f"| 两者共有组件数 | {shared} | - |",
        f"| 仅实验微调子集组件数 | {exp_only} | - |",
        f"| 仅MD预训练组件数 | {md_only} | - |",
    ]
    if merged_records is not None:
        lines.extend([
            "",
            f"合并文件记录数为 **{merged_records:,}**；原始两来源之和为 "
            f"**{exp_records + md_records:,}**。",
        ])
    lines.extend([
        "",
        "## 完整清单",
        "",
        "完整可筛选表见 [molecule_inventory.csv](molecule_inventory.csv)。"
        "其中两个“配方出现次数”是该分子在相应来源中出现的配方数，并非分子数。",
        "发布版生成器的 62 溶剂 + 2 盐固定词表与本清单的结构对应关系见 "
        "[released_vocabulary_comparison_zh.md](released_vocabulary_comparison_zh.md)。",
        "",
        "| 类别 | 名称 | SMILES | 实验微调出现配方数 | MD预训练出现配方数 | 来源分类 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for row in rows:
        safe_smiles = row["SMILES"].replace("|", "\\|")
        safe_name = row["名称"].replace("|", "\\|")
        lines.append(
            f"| {row['类别']} | {safe_name} | `{safe_smiles}` | "
            f"{row['实验微调子集_配方出现次数']} | "
            f"{row['MD预训练_配方出现次数']} | {row['来源分类']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@lru_cache(maxsize=None)
def canonical_smiles(smiles: str) -> str:
    """Use RDKit when available; otherwise retain the raw input representation."""
    if Chem is None:
        return smiles
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def collect_canonical_components(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, set[str]]]:
    result: dict[tuple[str, str], dict[str, set[str]]] = {}
    for record in records:
        for key, role in (("solvents", "溶剂"), ("salts", "盐")):
            for component in record.get(key, []):
                raw = str(component.get("smiles", "")).strip()
                if not raw:
                    raise ValueError(f"Missing SMILES in {key}: {component!r}")
                entry = result.setdefault((role, canonical_smiles(raw)), {
                    "名称": set(), "原始SMILES": set(),
                })
                name = str(component.get("name", "")).strip()
                if name:
                    entry["名称"].add(name)
                entry["原始SMILES"].add(raw)
    return result


def read_vocabulary(root: Path) -> dict[tuple[str, str], dict[str, set[str]]]:
    result: dict[tuple[str, str], dict[str, set[str]]] = {}
    for filename, role in (("solvs.json", "溶剂"), ("salts.json", "盐")):
        path = root / "bamboo_mixer" / "generator" / filename
        with path.open(encoding="utf-8") as handle:
            components = json.load(handle)
        for component in components:
            raw = str(component["smiles"]).strip()
            entry = result.setdefault((role, canonical_smiles(raw)), {
                "名称": set(), "原始SMILES": set(),
            })
            entry["名称"].add(str(component["name"]).strip())
            entry["原始SMILES"].add(raw)
    return result


def write_vocab_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["类别", "词表名称", "词表SMILES", "规范化SMILES", "实验微调子集", "MD预训练"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_vocabulary_comparison(output: Path, root: Path,
                                exp_records: list[dict[str, Any]],
                                md_records: list[dict[str, Any]]) -> None:
    """Compare the released 62-solvent/2-salt generation vocabulary to data."""
    vocabulary = read_vocabulary(root)
    exp = collect_canonical_components(exp_records)
    md = collect_canonical_components(md_records)
    raw_exp = collect(exp_records)
    raw_md = collect(md_records)

    vocab_rows: list[dict[str, str]] = []
    for (role, canonical), entry in sorted(vocabulary.items()):
        vocab_rows.append({
            "类别": role,
            "词表名称": " | ".join(sorted(entry["名称"])),
            "词表SMILES": " | ".join(sorted(entry["原始SMILES"])),
            "规范化SMILES": canonical,
            "实验微调子集": "是" if (role, canonical) in exp else "否",
            "MD预训练": "是" if (role, canonical) in md else "否",
        })
    write_vocab_csv(output / "released_vocabulary_comparison.csv", vocab_rows)

    outside_rows: list[dict[str, str]] = []
    for role, canonical in sorted((set(exp) | set(md)) - set(vocabulary)):
        exp_entry, md_entry = exp.get((role, canonical)), md.get((role, canonical))
        names = sorted((exp_entry or {"名称": set()})["名称"] |
                       (md_entry or {"名称": set()})["名称"])
        raw = sorted((exp_entry or {"原始SMILES": set()})["原始SMILES"] |
                     (md_entry or {"原始SMILES": set()})["原始SMILES"])
        outside_rows.append({
            "类别": role,
            "名称": " | ".join(names),
            "原始SMILES": " | ".join(raw),
            "规范化SMILES": canonical,
            "实验微调子集": "是" if exp_entry else "否",
            "MD预训练": "是" if md_entry else "否",
        })
    outside_path = output / "data_components_outside_released_vocabulary.csv"
    with outside_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["类别", "名称", "原始SMILES", "规范化SMILES", "实验微调子集", "MD预训练"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outside_rows)

    def cardinality(items: dict[tuple[str, str], Any], role: str) -> int:
        return sum(key[0] == role for key in items)

    def overlap(a: dict[tuple[str, str], Any], b: dict[tuple[str, str], Any], role: str) -> int:
        return sum(key[0] == role for key in set(a) & set(b))

    lines = [
        "# 发布版 Bamboo 词表与训练配方成分的关系",
        "",
        "## 结论",
        "",
        "发布版 Bamboo 生成器的内部候选空间是 **62 个溶剂 + 2 个盐**，",
        "不是整个微调集或 MD 集中出现过的所有成分。生成时 decoder 只能从该固定词表选择组分。",
        "",
        "下表按 RDKit 规范化 SMILES 比较（若运行环境没有 RDKit，则退化为原始 SMILES 比较）。",
        "因此可区分真实的结构不一致和仅由 SMILES 写法不同造成的表面不一致。",
        "",
        "| 类别 | 发布版词表 | 实验微调子集 | 词表∩实验 | MD预训练 | 词表∩MD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in ("溶剂", "盐"):
        lines.append(
            f"| {role} | {cardinality(vocabulary, role)} | {cardinality(exp, role)} | "
            f"{overlap(vocabulary, exp, role)} | {cardinality(md, role)} | "
            f"{overlap(vocabulary, md, role)} |"
        )
    lines.extend([
        "",
        f"共有 **{len(outside_rows)}** 个训练数据成分不在发布版词表中；"
        "它们可以参与预测器训练，却不能由发布版生成器直接生成。",
        "",
        "## 词表外训练成分",
        "",
        "完整表见 [data_components_outside_released_vocabulary.csv](data_components_outside_released_vocabulary.csv)。",
        "",
        "| 类别 | 名称 | 实验微调子集 | MD预训练 |",
        "| --- | --- | --- | --- |",
    ])
    for row in outside_rows:
        safe_name = row["名称"].replace("|", "\\|")
        lines.append(
            f"| {row['类别']} | {safe_name} | "
            f"{row['实验微调子集']} | {row['MD预训练']} |"
        )
    lines.extend([
        "",
        "## 发布版词表逐项映射",
        "",
        "完整表见 [released_vocabulary_comparison.csv](released_vocabulary_comparison.csv)。",
    ])
    (output / "released_vocabulary_comparison_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bamboo-root", type=Path,
                        default=Path("/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("run/bamboo_mixer_data_inventory"))
    args = parser.parse_args()

    root = args.bamboo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    exp_records = read_records(root, SOURCE_FILES["实验微调子集"])
    md_records = read_records(root, SOURCE_FILES["MD预训练"])
    rows = make_rows(collect(exp_records), collect(md_records))

    merged_records: int | None = None
    try:
        merged_records = len(read_records(root, MERGED_FILE))
    except FileNotFoundError:
        pass

    write_csv(output / "molecule_inventory.csv", rows)
    write_csv(output / "finetune_experimental_molecules.csv",
              [row for row in rows if row["实验微调子集_配方出现次数"]])
    write_csv(output / "md_pretrain_molecules.csv",
              [row for row in rows if row["MD预训练_配方出现次数"]])
    write_markdown(output / "molecule_inventory_zh.md", rows, len(exp_records),
                   len(md_records), merged_records)
    write_vocabulary_comparison(output, root, exp_records, md_records)
    audit = {
        "bamboo_root": str(root),
        "experimental_finetune_records": len(exp_records),
        "md_pretrain_records": len(md_records),
        "merged_finetune_records": merged_records,
        "raw_record_sum": len(exp_records) + len(md_records),
        "unique_component_rows": len(rows),
        "vocabulary_comparison_smiles_normalization": "RDKit canonical SMILES" if Chem else "raw SMILES fallback",
    }
    (output / "inventory_provenance.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} molecular component rows to {output}")


if __name__ == "__main__":
    main()
