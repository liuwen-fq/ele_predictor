#!/usr/bin/env python3
"""Summarize controlled joint-condition formula-generation experiments.

The report focuses on proximity to the requested conductivity *and* anion-ratio
condition.  It deliberately does not generate runtime or one-dimensional
target-hit-rate figures: those quantities are not sufficient for judging a
joint conditional generator.
"""

import argparse
import csv
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


GROUP_LABELS = {
    "G0": "G0 released Bamboo generator",
    "G0-Q": "G0-Q fixed-Bamboo-vocabulary property-only control",
    "G0-M": "G0-M fixed-Bamboo-vocabulary mechanism condition",
    "G0-MA": "G0-MA fixed-Bamboo-vocabulary interaction auxiliary supervision",
    "G0-MA-shuffle": "G0-MA-shuffle fixed-Bamboo-vocabulary shuffled interaction auxiliary supervision",
    "G0-MA-sigma": "G0-MA-sigma fixed-Bamboo-vocabulary I_sigma-only auxiliary supervision",
    "G0-MA-r": "G0-MA-r fixed-Bamboo-vocabulary I_r-only auxiliary supervision",
    "G0-MA-joint": "G0-MA-joint fixed-Bamboo-vocabulary I_sigma+I_r auxiliary supervision",
    "G0-I": "G0-I Bamboo-vocabulary interaction coverage",
    "G1": "G1 random/diverse pool",
    "G2-b": "G2-b fixed Bamboo vocabulary",
    "G2-b-Q": "G2-b-Q matched random control",
    "G2-b-I": "G2-b-I interaction-coverage selection",
    "G1-Q": "G1-Q matched random control",
    "G1-I": "G1-I interaction-coverage selection",
    "G2-Q": "G2-Q matched random control",
    "G2-M": "G2-M matched factual mechanism condition",
    "G2-M-shuffle": "G2-M-shuffle within-cell mechanism-label shuffle",
    "G2-MA": "G2-MA factual interaction-residual auxiliary supervision",
    "G2-MA-shuffle": "G2-MA-shuffle within-cell shuffled interaction-residual supervision",
    "G2-MA-sigma": "G2-MA-sigma I_sigma-only auxiliary supervision",
    "G2-MA-r": "G2-MA-r I_r-only auxiliary supervision",
    "G2-MA-joint": "G2-MA-joint I_sigma+I_r auxiliary supervision",
    "G2-I": "G2-I interaction-coverage selection",
    "G2": "G2 explain-guided pool",
    "G3": "G3 robust re-rank (G2)",
}
GROUPS = tuple(GROUP_LABELS)
COMPARISON_ORDER = (
    "G0", "G0-Q", "G0-M", "G0-MA", "G0-MA-shuffle", "G0-MA-sigma", "G0-MA-r", "G0-MA-joint", "G0-I",
    "G2-b", "G2-b-Q", "G2-b-I",
    "G1", "G1-Q", "G1-I",
    "G2", "G2-Q", "G2-M", "G2-M-shuffle", "G2-MA", "G2-MA-shuffle", "G2-MA-sigma", "G2-MA-r", "G2-MA-joint", "G2-I",
)
MAX_TRUSTED_CONDUCTIVITY = 100.0
# A joint error must first put σ (mS/cm) and r (dimensionless) on comparable
# scales.  They are defaults only: the command line makes the evaluation
# window explicit so that ±0.05 and ±0.10 r analyses cannot be conflated.
JOINT_SIGMA_SCALE = 2.0
JOINT_ANION_RATIO_SCALE = 0.10
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density_blue_to_red", ["#2563eb", "#dc2626"])
DENSITY_CMAP.set_bad("white")
GROUP_DESIGNS = {
    "G0": {
        "source": "Bamboo-Mixer 发布的 diffusion + decoder checkpoint；固定 Bamboo 62 溶剂、2 盐词表",
        "goal": "作为不使用本研究候选池筛选与解释约束的基线，衡量原版生成器的条件输出分布。",
    },
    "G0-I": {
        "source": "固定 Bamboo 62 溶剂与双盐组成先验；当前冻结 oracle 标注的 synthetic 配方在每个 (σ、r、溶剂数) 格内按 σ 解释交互低/中/高覆盖抽取后，重训 diffusion + decoder",
        "goal": "与发布版 G0 并列展示 Bamboo 词表下交互覆盖训练的端到端结果。注意：它不是在发布 G0 checkpoint 上继续训练，而是当前流程重训的交互覆盖组。",
    },
    "G0-Q": {
        "source": "固定 Bamboo 62 溶剂、双盐组成先验和当前冻结 oracle 的既有 30000 条 Q synthetic 数据；扩散条件仅为 (σ、r)",
        "goal": "作为 G0-M/G0-MA 的同数据、同词表且无机制输入或机制辅助监督对照。注意：它不是发布 Bamboo 的 G0 checkpoint。",
    },
    "G0-M": {
        "source": "复用 G0-Q 的相同 30000 条 synthetic 配方、σ/r 标签和固定 Bamboo 词表；扩散条件扩展为 (σ、r、Iσ、Ir)，decoder 与 G0-Q 共享",
        "goal": "检验真实公式—交互配对在固定 Bamboo 词表、当前生成流程下是否降低联合条件误差。它不等同于在发布 G0 checkpoint 上增加机制输入。",
    },
    "G0-MA": {
        "source": "复用 G0-Q 的相同 30000 条 synthetic 配方、σ/r 标签、固定 Bamboo 词表和当前冻结 oracle；diffusion 的生成条件仍仅为二维 (σ、r)，训练时以真实 [Iσ⊥、Ir⊥] 交互残差作为 denoising latent 的辅助监督，decoder 与 G0-Q 保持一致",
        "goal": "检验在固定 Bamboo 词表与当前 oracle 下，真实公式—交互残差配对能否通过训练期辅助任务降低联合条件生成误差；推理时不输入或采样交互标签，因此不等同于 G0-M 的四维机制条件。",
    },
    "G0-MA-shuffle": {
        "source": "与 G0-MA 使用完全相同的 fixed-Bamboo 30,000 条 synthetic 配方、σ/r 标签、词表、训练规模、二维生成条件和共享 decoder；仅在相同 (σ、r、溶剂数) 格内打乱完整 [Iσ⊥、Ir⊥] 辅助标签配对",
        "goal": "作为 G0-MA 的反事实对照：只有 G0-MA 同时优于 G0-Q 与该组，才支持固定 Bamboo 词表下真实公式—交互残差配对的正向作用。",
    },
    "G1": {
        "source": "从分子预训练数据的 20000 个候选中随机/多样性抽取 62 个分子；重新训练同规格 diffusion + decoder",
        "goal": "检验仅更换为随机且结构多样的分子库，能否改善目标 σ 与阴离子比例的生成结果。",
    },
    "G2": {
        "source": "从相同 20000 个候选中按片段贡献和分子物性软约束筛出 62 个分子；重新训练同规格 diffusion + decoder",
        "goal": "检验解释导出的结构与物性约束是否能将生成分布推向更可信、更接近目标的配方区域。",
    },
    "G2-b": {
        "source": "固定使用 Bamboo-Mixer 原始 62 溶剂、2 盐词表；在当前 oracle 与生成训练流程下重新训练 diffusion + decoder",
        "goal": "作为固定 Bamboo 词表消融，与 G2 的解释/物性约束词表比较；不执行 G2 选词步骤。",
    },
    "G2-b-Q": {
        "source": "固定 Bamboo 62 溶剂、1–6 溶剂经验分布与固定双盐；从同一 2×oracle 标注候选中按 (σ、r、溶剂数) 配额随机抽取 30000 条，重训 diffusion + decoder",
        "goal": "作为交互覆盖训练的严格随机对照，隔离候选池扩大与配额保持本身的影响。",
    },
    "G2-b-I": {
        "source": "与 G2-b-Q 使用同一候选构造、数量和 (σ、r、溶剂数) 配额；仅在每个格子内按 σ 解释交互低/中/高机制覆盖抽取，重训 diffusion + decoder",
        "goal": "检验模型内交互机制覆盖能否在不改变目标条件分布下改善生成多样性与条件一致性。",
    },
    "G1-Q": {
        "source": "G1 随机/多样性 62 溶剂词表；从同一 2×oracle 标注候选中按 (σ、r、溶剂数) 配额随机抽取 30000 条，重训 diffusion + decoder",
        "goal": "作为 G1-I 的严格随机对照，隔离候选池扩大、配额保持与交互覆盖选择的影响。",
    },
    "G1-I": {
        "source": "与 G1-Q 共享随机/多样性词表、候选构造、数量与 (σ、r、溶剂数) 配额；仅在每格中保持 σ 解释交互低/中/高覆盖，重训 diffusion + decoder",
        "goal": "检验交互覆盖在随机/多样性词表中是否改善联合条件可行性。",
    },
    "G2-Q": {
        "source": "G2 解释/物性软约束 62 溶剂词表；从同一 2×oracle 标注候选中按 (σ、r、溶剂数) 配额随机抽取 30000 条，重训 diffusion + decoder",
        "goal": "作为 G2-I 的严格随机对照，隔离解释筛词与交互覆盖训练的作用。",
    },
    "G2-I": {
        "source": "与 G2-Q 共享解释/物性软约束词表、候选构造、数量与 (σ、r、溶剂数) 配额；仅在每格中保持 σ 解释交互低/中/高覆盖，重训 diffusion + decoder",
        "goal": "检验交互覆盖能否与解释导向词表协同改善联合条件可行性。",
    },
    "G2-M": {
        "source": "逐条复用 G2-Q 的相同 30000 条 synthetic 配方、BOW、σ/r 标签和解释交互代理；仅将 diffusion 条件由 (σ、r) 扩展为 (σ、r、Iσ、Ir)，并复用 G2-Q decoder",
        "goal": "检验真实的公式—双任务交互配对是否有助于有限容量扩散模型学习联合条件配方流形。",
    },
    "G2-M-shuffle": {
        "source": "与 G2-M 使用完全相同的配方、σ/r 标签、词表、机制边际分布和共享 decoder；仅在相同 (σ、r、溶剂数) 格内打乱完整 [Iσ, Ir] 配对",
        "goal": "作为机制配对的反事实对照：若 G2-M 优于该组，差异可归因于真实公式—机制对应，而非机制维度或数据规模本身。",
    },
    "G2-MA": {
        "source": "逐条复用 G2-Q 的相同 30000 条 synthetic 配方、BOW、σ/r 标签与解释头交互残差；diffusion 的生成条件保持二维 (σ、r)，训练时以真实 [Iσ⊥、Ir⊥] 作为 denoising latent 的辅助监督，decoder 与 G2-Q 保持一致",
        "goal": "检验真实的公式—双任务交互残差配对能否在不增加推理期条件的前提下，改善联合条件配方生成。",
    },
    "G2-MA-shuffle": {
        "source": "与 G2-MA 使用完全相同的配方、σ/r 标签、词表、训练规模、生成条件与 decoder；仅在相同 (σ、r、溶剂数) 格内打乱完整 [Iσ⊥、Ir⊥] 辅助标签配对",
        "goal": "作为交互残差辅助监督的反事实对照：只有 G2-MA 同时优于 G2-Q 与该组，才支持真实公式—交互残差对应而非额外损失本身带来收益。",
    },
    "G2-MA-sigma": {
        "source": "复用 G2-Q 的固定 synthetic 配方、二维 (σ、r) 条件、词表与 decoder；训练期只监督 Iσ⊥ 残差。",
        "goal": "作为 σ-only 消融，判断 Iσ 是否能独立改善条件生成，且不受 Ir 标签影响。",
    },
    "G2-MA-r": {
        "source": "复用 G2-Q 的固定 synthetic 配方、二维 (σ、r) 条件、词表与 decoder；训练期只监督 Ir⊥ 残差。",
        "goal": "作为 r-only 消融，检验 Ir 是否含有稳定且可用于生成的机制信号。",
    },
    "G2-MA-joint": {
        "source": "复用 G2-Q 的固定 synthetic 配方、二维 (σ、r) 条件、词表与 decoder；训练期联合监督 [Iσ⊥、Ir⊥]。",
        "goal": "作为显式 σ+r MA 组，与两个单任务消融共同判断标签质量和双任务冲突。",
    },
    "G0-MA-sigma": {
        "source": "复用 G0-Q 的固定 Bamboo 词表 synthetic 配方、二维 (σ、r) 条件与共享 decoder；训练期只监督 Iσ⊥。",
        "goal": "检验固定 Bamboo 词表下 Iσ 的独立辅助价值。",
    },
    "G0-MA-r": {
        "source": "复用 G0-Q 的固定 Bamboo 词表 synthetic 配方、二维 (σ、r) 条件与共享 decoder；训练期只监督 Ir⊥。",
        "goal": "检验固定 Bamboo 词表下 Ir 是否造成 r 或联合条件的负迁移。",
    },
    "G0-MA-joint": {
        "source": "复用 G0-Q 的固定 Bamboo 词表 synthetic 配方、二维 (σ、r) 条件与共享 decoder；训练期联合监督 [Iσ⊥、Ir⊥]。",
        "goal": "作为固定 Bamboo 词表下显式 σ+r MA 对照。",
    },
    "G3": {
        "source": "复用 G2 生成器与候选池，在温度—浓度网格上进行鲁棒重排序",
        "goal": "检验后处理的鲁棒排序能否提高有限实验预算下候选的可用性。",
    },
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solvent_meta(root, group):
    pool = root / group / "candidate_pool"
    if group == "G3":
        pool = root / "G2" / "candidate_pool"
    return json.loads((pool / "solvents.json").read_text())


def trusted_record(row):
    """Return whether a generated record is safe for quantitative comparison."""
    value = row.get("conductivity_pred_ms_cm")
    return (row.get("prediction_status", "trusted") == "trusted"
            and value is not None
            and np.isfinite(float(value))
            and 0.0 <= float(value) <= MAX_TRUSTED_CONDUCTIVITY)


def formula_label(row, solvents, group, threshold=.02):
    """Use released vocabulary names for G0/G2-b and SMILES otherwise."""
    bow = np.asarray(row["bow"], dtype=float)
    terms = []
    for index in np.where(bow[:len(solvents)] > threshold)[0]:
        identity = (solvents[index]["name"] if group in {"G0", "G0-Q", "G0-M", "G0-MA", "G0-MA-shuffle", "G0-MA-sigma", "G0-MA-r", "G0-MA-joint", "G0-I", "G2-b", "G2-b-Q", "G2-b-I"}
                    else solvents[index]["smiles"])
        terms.append(f"{identity} ({bow[index]:.2f})")
    if not terms:
        index = int(np.argmax(bow[:len(solvents)]))
        identity = (solvents[index]["name"] if group in {"G0", "G0-Q", "G0-M", "G0-MA", "G0-MA-shuffle", "G0-MA-sigma", "G0-MA-r", "G0-MA-joint", "G0-I", "G2-b", "G2-b-Q", "G2-b-I"}
                    else solvents[index]["smiles"])
        terms.append(f"{identity} ({bow[index]:.2f})")
    return " + ".join(terms)


def summarize_group(root, group, target, selection_budget):
    generated = root / group / "generated" / "generated_formulas.json"
    if not generated.exists():
        return None
    records = json.loads(generated.read_text())
    if not records:
        return None
    solvents = solvent_meta(root, group)
    trusted_ids = np.asarray([
        index for index, row in enumerate(records)
        if trusted_record(row)
    ])
    if not len(trusted_ids):
        return None
    values = np.asarray([float(records[index]["conductivity_pred_ms_cm"])
                         for index in trusted_ids])
    ranked_ids = np.asarray([
        index for index in trusted_ids
        if records[index].get("candidate_rank_score") is not None
        and np.isfinite(float(records[index]["candidate_rank_score"]))
    ])
    rank = np.asarray([float(records[index]["candidate_rank_score"])
                       for index in ranked_ids])
    selected_positions = np.argsort(-rank)[:min(selection_budget, len(ranked_ids))]
    selected_ids = ranked_ids[selected_positions]
    selected_values = np.asarray([
        float(records[index]["conductivity_pred_ms_cm"])
        for index in selected_ids
    ])
    trusted_records = [records[index] for index in trusted_ids]
    solvent_counts, dominant = [], []
    formula_keys = []
    for row in trusted_records:
        bow = np.asarray(row["bow"], dtype=float)
        active = np.where(bow[:len(solvents)] > 0.02)[0]
        solvent_counts.append(int(len(active)))
        dominant.append(solvents[int(np.argmax(bow[:len(solvents)]))]["name"])
        formula_keys.append("+".join(
            f"{solvents[i]['name']}:{bow[i]:.2f}" for i in active))
    robust_min = [row.get("robust_min_ms_cm") for row in trusted_records]
    predicted_ratios = np.asarray([
        float(row["anion_ratio_pred"]) for row in trusted_records
        if row.get("anion_ratio_pred") is not None
        and np.isfinite(float(row["anion_ratio_pred"]))
    ])
    condition = float(records[0].get("anion_ratio_condition", .5))
    selected_ratios = np.asarray([
        float(records[index]["anion_ratio_pred"]) for index in selected_ids
        if records[index].get("anion_ratio_pred") is not None
        and np.isfinite(float(records[index]["anion_ratio_pred"]))
    ])
    paired_values, paired_ratios = zip(*[
        (float(row["conductivity_pred_ms_cm"]), float(row["anion_ratio_pred"]))
        for row in trusted_records
        if row.get("anion_ratio_pred") is not None
        and np.isfinite(float(row["anion_ratio_pred"]))
    ]) if len(predicted_ratios) else ([], [])
    paired_values = np.asarray(paired_values, dtype=float)
    paired_ratios = np.asarray(paired_ratios, dtype=float)
    selected_pairs = [
        (float(records[index]["conductivity_pred_ms_cm"]),
         float(records[index]["anion_ratio_pred"]))
        for index in selected_ids
        if records[index].get("anion_ratio_pred") is not None
        and np.isfinite(float(records[index]["anion_ratio_pred"]))
    ]
    selected_pair_values = np.asarray([item[0] for item in selected_pairs], dtype=float)
    selected_pair_ratios = np.asarray([item[1] for item in selected_pairs], dtype=float)
    abs_k_error = np.abs(values - target)
    abs_r_error = np.abs(paired_ratios - condition)
    selected_abs_k_error = np.abs(selected_values - target)
    selected_abs_r_error = np.abs(selected_pair_ratios - condition)
    normalized_sigma_error = (paired_values - target) / JOINT_SIGMA_SCALE
    normalized_ratio_error = (
        (paired_ratios - condition) / JOINT_ANION_RATIO_SCALE)
    selected_normalized_sigma_error = (
        (selected_pair_values - target) / JOINT_SIGMA_SCALE)
    selected_normalized_ratio_error = (
        (selected_pair_ratios - condition) / JOINT_ANION_RATIO_SCALE)
    def rate(mask):
        return float(np.mean(mask)) if len(mask) else ""
    def scalar_metric(values, fn):
        return float(fn(values)) if len(values) else ""
    oracle_sources = sorted({str(row.get("oracle_source") or "current_dual_task_oracle")
                             for row in records})
    return {
        "group": group,
        "group_label": GROUP_LABELS[group],
        "records": records,
        "trusted_records": trusted_records,
        "trusted_ids": trusted_ids,
        "solvents": solvents,
        "values": values,
        "selected_ids": selected_ids,
        "summary": {
            "group": group,
            "policy": ("released Bamboo generator checkpoint" if group == "G0" else
                       "fixed-Bamboo-vocabulary property-only control" if group == "G0-Q" else
                       "fixed-Bamboo-vocabulary explicit mechanism condition" if group == "G0-M" else
                       "fixed-Bamboo-vocabulary factual interaction-residual auxiliary supervision" if group == "G0-MA" else
                       "fixed-Bamboo-vocabulary within-cell shuffled interaction-residual auxiliary supervision" if group == "G0-MA-shuffle" else
                       "random/diverse 62-solvent pool" if group == "G1" else
                       "Bamboo-vocabulary sigma-interaction coverage selection" if group == "G0-I" else
                       "fixed Bamboo 62-solvent vocabulary ablation" if group == "G2-b" else
                       "matched random synthetic-data selection control" if group in {"G2-b-Q", "G1-Q", "G2-Q"} else
                       "factual [I_sigma, I_r] explicit diffusion condition" if group == "G2-M" else
                       "within-cell shuffled [I_sigma, I_r] diffusion condition" if group == "G2-M-shuffle" else
                       "factual [I_sigma_perp, I_r_perp] auxiliary supervision with 2D (sigma, r) generation condition" if group == "G2-MA" else
                       "within-cell shuffled [I_sigma_perp, I_r_perp] auxiliary supervision with 2D (sigma, r) generation condition" if group == "G2-MA-shuffle" else
                       "within-condition sigma-interaction coverage selection" if group in {"G2-b-I", "G1-I", "G2-I"} else
                       "fragment/property guided 62-solvent pool" if group == "G2" else
                       "G2 generator + robustness/explanation re-ranking"),
            "n_sampled": len(records),
            "n_trusted": len(trusted_ids),
            "n_out_of_domain": len(records) - len(trusted_ids),
            "target_mS_cm": target,
            "anion_ratio_condition": condition,
            "oracle_source": "; ".join(oracle_sources),
            "all_hit_count": int(np.sum(values >= target)),
            "all_hit_rate": float(np.mean(values >= target)),
            "kappa_mae_mS_cm": float(np.mean(abs_k_error)),
            "kappa_rmse_mS_cm": float(np.sqrt(np.mean((values - target) ** 2))),
            "kappa_mape_pct": float(np.mean(abs_k_error / max(abs(target), 1e-8)) * 100),
            "kappa_median_ae_mS_cm": float(np.median(abs_k_error)),
            "kappa_within_pm1": rate(abs_k_error <= 1.0),
            "kappa_within_pm2": rate(abs_k_error <= 2.0),
            "kappa_within_pm5": rate(abs_k_error <= 5.0),
            "near_target_rate_pm2": rate(abs_k_error <= 2.0),
            "mean_abs_target_gap_mS_cm": float(np.mean(abs_k_error)),
            "selected_budget": len(selected_ids),
            "selected_hit_count": int(np.sum(selected_values >= target)),
            "selected_hit_rate": float(np.mean(selected_values >= target)),
            "selected_mean_mS_cm": float(np.mean(selected_values)),
            "selected_min_mS_cm": float(np.min(selected_values)),
            "selected_kappa_mae_mS_cm": float(np.mean(selected_abs_k_error)),
            "selected_kappa_rmse_mS_cm": float(np.sqrt(np.mean((selected_values - target) ** 2))),
            "selected_kappa_mape_pct": float(np.mean(selected_abs_k_error / max(abs(target), 1e-8)) * 100),
            "selected_kappa_within_pm2": rate(selected_abs_k_error <= 2.0),
            "all_median_mS_cm": float(np.median(values)),
            "all_p90_mS_cm": float(np.percentile(values, 90)),
            "unique_compositions": len(set(formula_keys)),
            "mean_active_solvents": float(np.mean(solvent_counts)),
            "top_dominant_solvent": Counter(dominant).most_common(1)[0][0],
            "anion_ratio_pred_mae_from_condition": (
                float(np.mean(abs_r_error))
                if len(predicted_ratios) else ""
            ),
            "anion_ratio_pred_rmse_from_condition": (
                float(np.sqrt(np.mean((paired_ratios - condition) ** 2)))
                if len(predicted_ratios) else ""
            ),
            "anion_ratio_within_pm0.05": rate(abs_r_error <= .05),
            "anion_ratio_pred_within_pm0.1": (
                rate(abs_r_error <= .1)
                if len(predicted_ratios) else ""
            ),
            "joint_normalized_mae": (
                float(np.mean((np.abs(normalized_sigma_error)
                               + np.abs(normalized_ratio_error)) / 2.0))
                if len(paired_values) else ""
            ),
            "joint_normalized_rmse": (
                float(np.sqrt(np.mean((np.square(normalized_sigma_error)
                                       + np.square(normalized_ratio_error)) / 2.0)))
                if len(paired_values) else ""
            ),
            "selected_anion_ratio_pred_mae_from_condition": (
                scalar_metric(selected_abs_r_error, np.mean)
                if len(selected_ratios) else ""
            ),
            "selected_anion_ratio_pred_rmse_from_condition": (
                scalar_metric(selected_pair_ratios - condition, lambda x: np.sqrt(np.mean(x ** 2)))
                if len(selected_ratios) else ""
            ),
            "selected_anion_ratio_within_pm0.05": rate(selected_abs_r_error <= .05),
            "selected_anion_ratio_pred_within_pm0.1": (
                rate(selected_abs_r_error <= .1)
                if len(selected_ratios) else ""
            ),
            "selected_joint_normalized_mae": (
                float(np.mean((np.abs(selected_normalized_sigma_error)
                               + np.abs(selected_normalized_ratio_error)) / 2.0))
                if len(selected_pair_values) else ""
            ),
            "selected_joint_normalized_rmse": (
                float(np.sqrt(np.mean((np.square(selected_normalized_sigma_error)
                                       + np.square(selected_normalized_ratio_error)) / 2.0)))
                if len(selected_pair_values) else ""
            ),
            "joint_within_kappa_pm2_ratio_pm0.05": rate(
                (np.abs(paired_values - target) <= 2.0) & (abs_r_error <= .05)),
            "joint_within_kappa_pm2_ratio_pm0.1": rate(
                (np.abs(paired_values - target) <= 2.0) & (abs_r_error <= .1)),
            "selected_joint_within_kappa_pm2_ratio_pm0.05": rate(
                (np.abs(selected_pair_values - target) <= 2.0) & (selected_abs_r_error <= .05)),
            "selected_joint_within_kappa_pm2_ratio_pm0.1": rate(
                (np.abs(selected_pair_values - target) <= 2.0) & (selected_abs_r_error <= .1)),
            "g3_robust_min_median_mS_cm": (
                float(np.median([x for x in robust_min if x is not None]))
                if any(x is not None for x in robust_min) else ""),
        },
        "solvent_counts": np.asarray(solvent_counts),
    }


def timing_summary(root, group):
    path = root / group / "timings.csv"
    if not path.exists():
        return None, []
    rows = read_csv(path)
    stages = defaultdict(float)
    for row in rows:
        if row["detail"] == "executed":
            stages[row["stage"]] += float(row["elapsed_seconds"])
    core = ("candidate_pool", "vocabulary_embedding", "synthetic_formula_oracle",
            "write_train_configs", "train_diffusion", "train_decoder", "sample_and_score")
    summary = {"group": group, "total_executed_seconds": round(sum(stages.values()), 3)}
    for stage in core:
        summary[f"{stage}_seconds"] = round(stages.get(stage, 0.0), 3)
    return summary, rows


def comparison_order(results):
    """Use the requested paired layout: G0/G2-b above, G1/G2 below."""
    by_group = {result["group"]: result for result in results}
    return [by_group[group] for group in COMPARISON_ORDER if group in by_group] + [
        result for result in results if result["group"] not in COMPARISON_ORDER]


def save_heatmap(groups, out, target):
    """Compact group-by-conductivity-bin heatmap for distribution comparison."""
    groups = comparison_order(groups)
    edges = np.asarray([0, 5, 10, 15, 20, 25, 30, 40, np.inf])
    labels = ["0–5", "5–10", "10–15", "15–20", "20–25", "25–30", "30–40", "≥40"]
    matrix = np.asarray([
        np.histogram(result["values"], bins=edges)[0] / len(result["values"])
        for result in groups
    ])
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(groups)), labels=[result["group"] for result in groups])
    ax.set_xlabel(f"Predicted σ bin (mS/cm); target = {target:g}")
    ax.set_ylabel("Generation experiment")
    ax.set_title("Generated formula σ coverage")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{100 * value:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if value < .55 else "white")
    fig.colorbar(image, ax=ax, label="Fraction of trusted formulas")
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_target_window_metrics(groups, out):
    """2×2 per-group view of σ, ratio, and joint target-window coverage."""
    groups = comparison_order(groups)
    specs = [
        ("σ ±1", "kappa_within_pm1"),
        ("σ ±2", "kappa_within_pm2"),
        ("r ±0.05", "anion_ratio_within_pm0.05"),
        ("r ±0.10", "anion_ratio_pred_within_pm0.1"),
        ("joint\n±2, ±0.05", "joint_within_kappa_pm2_ratio_pm0.05"),
        ("joint\n±2, ±0.10", "joint_within_kappa_pm2_ratio_pm0.1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharey=True, squeeze=False)
    for ax, result in zip(axes.flat, groups):
        summary = result["summary"]
        values = [float(summary[key]) * 100 if summary[key] != "" else 0. for _, key in specs]
        bars = ax.bar(np.arange(len(specs)), values,
                      color=["#60a5fa", "#2563eb", "#a7f3d0", "#10b981", "#fbbf24", "#f97316"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(result["group"])
        ax.set_xticks(np.arange(len(specs)), labels=[name for name, _ in specs], fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.22)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share among trusted generated formulas (%)")
    fig.suptitle("Joint-condition target-window coverage", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_target_and_joint_error_metrics(groups, out):
    """Compare all trusted formulas on σ, r, and scale-normalized joint error."""
    groups = comparison_order(groups)
    metrics = [
        ("σ MAE", "kappa_mae_mS_cm", "mS/cm"),
        ("σ RMSE", "kappa_rmse_mS_cm", "mS/cm"),
        ("r MAE", "anion_ratio_pred_mae_from_condition", "ratio"),
        ("r RMSE", "anion_ratio_pred_rmse_from_condition", "ratio"),
        ("Joint normalized MAE", "joint_normalized_mae", "normalized joint error"),
        ("Joint normalized RMSE", "joint_normalized_rmse", "normalized joint error"),
    ]
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706"]
    # Pair MAE and RMSE by task: σ, r, then the two-target joint error.
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 10.8), squeeze=False)
    group_names = [result["group"] for result in groups]
    for ax, (title, key, unit) in zip(axes.flat, metrics):
        values = [float(result["summary"][key]) for result in groups]
        bars = ax.bar(group_names, values, color=colors, alpha=.86)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.18 if max(values) else 1)
        ax.grid(axis="y", alpha=.22)
    fig.suptitle(
        f"All trusted generated formulas: separate and joint target error "
        f"(σ / {JOINT_SIGMA_SCALE:g} mS/cm, r / {JOINT_ANION_RATIO_SCALE:g})",
        y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
def save_selected_joint_error_metrics(groups, out):
    """Compare the equal-budget top candidates on each task and jointly."""
    groups = comparison_order(groups)
    metrics = [
        ("Top-100 σ MAE", "selected_kappa_mae_mS_cm", "mS/cm"),
        ("Top-100 σ RMSE", "selected_kappa_rmse_mS_cm", "mS/cm"),
        ("Top-100 r MAE", "selected_anion_ratio_pred_mae_from_condition", "ratio"),
        ("Top-100 r RMSE", "selected_anion_ratio_pred_rmse_from_condition", "ratio"),
        ("Top-100 joint normalized MAE", "selected_joint_normalized_mae", "normalized joint error"),
        ("Top-100 joint normalized RMSE", "selected_joint_normalized_rmse", "normalized joint error"),
    ]
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706"]
    group_names = [result["group"] for result in groups]
    # Keep MAE and RMSE for the same target on one row, so the progression
    # from σ, to r, to the joint score is readable vertically.
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 10.8), squeeze=False)
    for ax, (title, key, unit) in zip(axes.flat, metrics):
        values = [float(result["summary"][key]) for result in groups]
        bars = ax.bar(group_names, values, color=colors[:len(groups)], alpha=.86)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.18 if max(values) else 1)
        ax.grid(axis="y", alpha=.22)
    fig.suptitle(
        f"Equal-budget top candidates: separate and joint target error "
        f"(σ / {JOINT_SIGMA_SCALE:g} mS/cm, r / {JOINT_ANION_RATIO_SCALE:g})",
        y=.99)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_hit_rate(groups, out, target):
    x = np.arange(len(groups))
    all_rates = [r["summary"]["all_hit_rate"] * 100 for r in groups]
    selected_rates = [r["summary"]["selected_hit_rate"] * 100 for r in groups]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = .36
    bars_a = ax.bar(x - width / 2, all_rates, width, label="all sampled")
    bars_b = ax.bar(x + width / 2, selected_rates, width, label="equal selection budget")
    ax.bar_label(bars_a, labels=[f"{v:.1f}%" for v in all_rates], padding=3, fontsize=8)
    ax.bar_label(bars_b, labels=[f"{v:.1f}%" for v in selected_rates], padding=3, fontsize=8)
    ax.set_xticks(x, [r["group"] for r in groups])
    ax.set_ylim(0, max([10] + all_rates + selected_rates) * 1.22)
    ax.set_ylabel("Target-hit rate (%)")
    ax.set_title(f"Share of generated formulas reaching {target:g} mS/cm")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_formula_range(groups, out, target):
    groups = comparison_order(groups)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True,
                             squeeze=False)
    for ax, result in zip(axes.flat, groups):
        values = result["values"]
        counts = result["solvent_counts"]
        hit = values >= target
        ax.scatter(counts[~hit], values[~hit], s=18, alpha=.55, c="#6b7280", label="below target")
        ax.scatter(counts[hit], values[hit], s=20, alpha=.7, c="#dc2626", label="target hit")
        ax.axhline(target, color="#dc2626", lw=1, ls="--")
        ax.set_title(result["group"], pad=6)
        ax.set_xlabel("Active solvents")
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.grid(alpha=.2)
    axes[0][0].set_ylabel("Predicted σ (mS/cm)")
    axes[1][0].set_ylabel("Predicted σ (mS/cm)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .005))
    fig.suptitle("Generated formula complexity and target coverage", y=.985)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def top_near_target(result, target, limit=10):
    """Keep the closest unique solvent mixtures, not repeated diffusion samples."""
    unique = {}
    for record in result["trusted_records"]:
        label = formula_label(record, result["solvents"], result["group"])
        value = float(record["conductivity_pred_ms_cm"])
        candidate = (abs(value - target), value, record, label)
        if label not in unique or candidate[:2] < unique[label][:2]:
            unique[label] = candidate
    selected = sorted(unique.values(), key=lambda item: item[:2])[:limit]
    rows = []
    for rank, (distance, value, record, label) in enumerate(selected, 1):
        rows.append({
            "rank_by_abs_error_to_target": rank,
            "conductivity_pred_ms_cm": round(value, 4),
            "abs_difference_from_target_mS_cm": round(distance, 4),
            "solvent_formula_smiles_and_molar_ratio": label,
            "anion_ratio_condition": record.get("anion_ratio_condition"),
            "salt_molar_ratio": record.get("salt_molar_ratio"),
            "temperature_c": record.get("temperature_c"),
        })
    return rows


def save_top_near_target(result, out, target):
    rows = top_near_target(result, target)
    if not rows:
        return rows
    values = np.asarray([row["conductivity_pred_ms_cm"] for row in rows])
    labels = [textwrap.fill(row["solvent_formula_smiles_and_molar_ratio"], 60)
              for row in rows]
    fig, ax = plt.subplots(figsize=(14, 9.0))
    y = np.arange(len(rows))
    colors = ["#d97706" if value < target else "#2563eb" for value in values]
    ax.barh(y, values, color=colors, alpha=.82, height=.65)
    ax.axvline(target, color="#dc2626", ls="--", lw=1.4,
               label=f"target = {target:g} mS/cm")
    for pos, row in enumerate(rows):
        ax.text(values[pos] + .08, pos, f"{values[pos]:.2f}", va="center",
                fontsize=9)
    ax.set_yticks(y, labels=labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(target * 1.16, values.max() * 1.16))
    ax.set_xlabel("Frozen-oracle predicted σ (mS/cm)")
    ax.set_title(f"{result['group']}: 10 unique generated formulas closest to {target:g} mS/cm")
    ax.grid(axis="x", alpha=.25)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=.48, right=.96, top=.92, bottom=.08)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return rows


def conductivity_anion_density(result):
    """Fixed-axis density; normalizing by all samples exposes OOD loss."""
    pairs = [(float(row["conductivity_pred_ms_cm"]),
              float(row["anion_ratio_pred"]))
             for row in result["trusted_records"]
             if row.get("anion_ratio_pred") is not None
             and np.isfinite(float(row["anion_ratio_pred"]))]
    if not pairs:
        return None
    values, ratios = (np.asarray(item) for item in zip(*pairs))
    x_edges = np.linspace(0.0, MAX_TRUSTED_CONDUCTIVITY, 31)
    y_edges = np.linspace(0.0, 1.0, 26)
    counts, _, _ = np.histogram2d(values, ratios, bins=(x_edges, y_edges))
    return counts.T / len(result["records"]), x_edges, y_edges


def save_anion_ratio_distribution(result, out, target):
    """Density map of oracle-predicted conductivity and anion ratio."""
    density = conductivity_anion_density(result)
    if density is None:
        return False
    matrix, x_edges, y_edges = density
    matrix = np.ma.masked_less_equal(matrix, 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.pcolormesh(x_edges, y_edges, matrix, cmap=DENSITY_CMAP,
                          shading="auto")
    colorbar = fig.colorbar(image, ax=ax, pad=.02)
    colorbar.set_label("Density (fraction of all generated formulas per bin)")
    ax.axvline(target, color="#dc2626", ls="--", lw=1.2,
               label=f"target = {target:g} mS/cm")
    condition = result["trusted_records"][0].get("anion_ratio_condition")
    if condition is not None:
        ax.axhline(float(condition), color="white", ls=":", lw=1.4,
                   label=f"generation condition = {float(condition):g}")
    ax.set_xlim(0, MAX_TRUSTED_CONDUCTIVITY)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frozen-oracle predicted σ (mS/cm)")
    ax.set_ylabel("Frozen-oracle predicted anion ratio")
    ax.set_title(f"{result['group']}: σ–anion-ratio density")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .97,
            f"trusted / sampled = {len(result['trusted_records'])} / {len(result['records'])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_anion_ratio_density_comparison(results, out, target):
    """Shared-scale 2×2 panels in the paired G0/G2-b, G1/G2 layout."""
    prepared = [(result, conductivity_anion_density(result))
                for result in comparison_order(results)]
    prepared = [(result, item) for result, item in prepared if item is not None]
    if not prepared:
        return False
    vmax = max(float(item[0].max()) for _, item in prepared)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True, sharey=True,
                             squeeze=False)
    image = None
    for ax, (result, (matrix, x_edges, y_edges)) in zip(axes.flat, prepared):
        matrix = np.ma.masked_less_equal(matrix, 0.0)
        image = ax.pcolormesh(x_edges, y_edges, matrix, cmap=DENSITY_CMAP,
                              shading="auto", vmin=0, vmax=vmax)
        ax.axvline(target, color="#dc2626", ls="--", lw=1.1)
        condition = result["trusted_records"][0].get("anion_ratio_condition")
        if condition is not None:
            ax.axhline(float(condition), color="white", ls=":", lw=1.2)
        ax.set_title(result["group"])
        ax.set_xlim(0, MAX_TRUSTED_CONDUCTIVITY)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.18)
        ax.set_xlabel("Predicted σ (mS/cm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted anion ratio")
    for ax in axes.flat[len(prepared):]:
        ax.axis("off")
    fig.subplots_adjust(left=.08, right=.86, bottom=.10, top=.90, wspace=.10, hspace=.14)
    colorbar_axis = fig.add_axes([.88, .20, .018, .54])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Density (fraction of all generated formulas per bin)")
    fig.suptitle("Generated σ–anion-ratio coverage (shared density scale)", y=.97)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_runtime(timings, out):
    stages = ("candidate_pool", "vocabulary_embedding", "synthetic_formula_oracle",
              "train_diffusion", "train_decoder", "sample_and_score")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(stages)))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(timings))
    for stage, color in zip(stages, colors):
        values = np.asarray([row.get(f"{stage}_seconds", 0.) for row in timings]) / 60
        ax.bar([row["group"] for row in timings], values, bottom=bottom, label=stage.replace("_", " "), color=color)
        bottom += values
    for i, value in enumerate(bottom):
        ax.text(i, value, f"{value:.1f} min", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Executed wall-clock time (min)")
    ax.set_title("Generation experiment runtime by stage")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |")
    return "\n".join(lines)


def display_metric_rows(rows):
    """Keep report tables readable while retaining full precision in the CSV."""
    formatted = []
    for row in rows:
        output = {}
        for key, value in row.items():
            if value == "":
                output[key] = "—"
            elif key.endswith(("_within_pm1", "_within_pm2", "_within_pm5",
                               "_within_pm0.05", "_within_pm0.1")) or key.startswith(("all_hit", "near_target")):
                output[key] = f"{float(value) * 100:.1f}%"
            elif isinstance(value, float):
                output[key] = f"{value:.3f}"
            else:
                output[key] = value
        formatted.append(output)
    return formatted


def main():
    global JOINT_SIGMA_SCALE, JOINT_ANION_RATIO_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default="run/generation_experiments")
    parser.add_argument("--target-conductivity", type=float, default=20.)
    parser.add_argument("--selection-budget", type=int, default=100)
    parser.add_argument("--joint-sigma-scale", type=float, default=JOINT_SIGMA_SCALE,
                        help="σ normalization scale for joint MAE/RMSE (mS/cm).")
    parser.add_argument("--joint-anion-ratio-scale", type=float,
                        default=JOINT_ANION_RATIO_SCALE,
                        help="r normalization scale for joint MAE/RMSE.")
    parser.add_argument("--groups", nargs="+", choices=GROUPS,
                        default=list(GROUPS),
                        help="Groups to summarize; defaults to every available group.")
    args = parser.parse_args()
    if args.joint_sigma_scale <= 0 or args.joint_anion_ratio_scale <= 0:
        parser.error("Joint normalization scales must be positive.")
    JOINT_SIGMA_SCALE = args.joint_sigma_scale
    JOINT_ANION_RATIO_SCALE = args.joint_anion_ratio_scale
    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    results = [summarize_group(root, group, args.target_conductivity, args.selection_budget)
               for group in args.groups]
    results = [x for x in results if x is not None]
    if not results:
        raise SystemExit(f"No generated_formulas.json found under {root}")
    results = comparison_order(results)
    summary = [x["summary"] for x in results]
    write_csv(root / "generation_experiment_summary.csv", summary)
    save_heatmap(results, root / "conductivity_coverage_heatmap.png", args.target_conductivity)
    save_target_window_metrics(results, root / "joint_target_window_metrics_2x2.png")
    save_target_and_joint_error_metrics(
        results, root / "target_joint_error_metrics_3x2.png")
    save_selected_joint_error_metrics(
        results, root / "selected_joint_error_metrics.png")
    save_formula_range(results, root / "generated_formula_range.png", args.target_conductivity)
    top_figures = []
    for result in results:
        group = result["group"]
        top_rows = save_top_near_target(
            result, root / f"top10_near_{args.target_conductivity:g}_mS_cm_{group}.png",
            args.target_conductivity)
        if top_rows:
            csv_path = root / f"top10_near_{args.target_conductivity:g}_mS_cm_{group}.csv"
            write_csv(csv_path, top_rows)
            top_figures.append((group, csv_path.name,
                                f"top10_near_{args.target_conductivity:g}_mS_cm_{group}.png"))
    density_comparison = root / "conductivity_anion_density_comparison.png"
    has_density_comparison = save_anion_ratio_density_comparison(
        results, density_comparison, args.target_conductivity)

    design_rows = [{"group": result["group"],
                    "生成器/候选池": GROUP_DESIGNS[result["group"]]["source"],
                    "研究目标": GROUP_DESIGNS[result["group"]]["goal"]}
                   for result in results]
    display_summary = display_metric_rows(summary)
    condition = float(results[0]["records"][0].get("anion_ratio_condition", .5))
    ratio_lower = float(results[0]["records"][0].get("anion_ratio_eligibility_min", 0.0))
    ratio_upper = float(results[0]["records"][0].get("anion_ratio_eligibility_max", 1.0))
    report = [
        "# 联合条件生成实验结果", "",
        f"共同采样条件：目标 σ **{args.target_conductivity:g} mS/cm**、条件阴离子比例 **{condition:g}**、"
        f"温度 25 °C、salt_molar_ratio 0.10。"
        + (f" 本实验额外启用 anion-ratio 筛选区间 **[{ratio_lower:g}, {ratio_upper:g}]**。"
           if (ratio_lower > 0.0 or ratio_upper < 1.0) else
           " 本实验不对生成结果施加 anion-ratio 后处理筛选。"),
        "",
        "## 研究目标与分组设计", "",
        markdown_table(design_rows, ["group", "生成器/候选池", "研究目标"]), "",
        "除发布 checkpoint 的 G0 外，各重训组均使用同一冻结 oracle、相同 synthetic 配方数量和同规格 diffusion/decoder。"
        "比较候选池时必须锁定组成数分布、盐采样、随机种子和训练预算；固定 Bamboo 词表组不应被误标为解释约束筛词组。",
        "",
        "## 全体可信生成配方：目标误差与窗口覆盖", "",
        markdown_table(display_summary, ["group", "n_sampled", "n_trusted", "n_out_of_domain", "kappa_mae_mS_cm", "kappa_rmse_mS_cm", "kappa_mape_pct", "kappa_within_pm1", "kappa_within_pm2", "kappa_within_pm5", "anion_ratio_pred_mae_from_condition", "anion_ratio_pred_rmse_from_condition", "anion_ratio_within_pm0.05", "anion_ratio_pred_within_pm0.1", "joint_normalized_mae", "joint_normalized_rmse", "joint_within_kappa_pm2_ratio_pm0.05", "joint_within_kappa_pm2_ratio_pm0.1"]),
        "",
        "## 前 100 个联合排序候选：目标误差与窗口覆盖", "",
        markdown_table(display_summary, ["group", "selected_budget", "selected_kappa_mae_mS_cm", "selected_kappa_rmse_mS_cm", "selected_kappa_mape_pct", "selected_kappa_within_pm2", "selected_anion_ratio_pred_mae_from_condition", "selected_anion_ratio_pred_rmse_from_condition", "selected_anion_ratio_within_pm0.05", "selected_anion_ratio_pred_within_pm0.1", "selected_joint_normalized_mae", "selected_joint_normalized_rmse", "selected_joint_within_kappa_pm2_ratio_pm0.05", "selected_joint_within_kappa_pm2_ratio_pm0.1", "unique_compositions", "mean_active_solvents"]),
        "",
        "### 参数与指标含义", "",
        "| 参数/指标 | 含义与使用意义 |",
        "| --- | --- |",
        f"| target σ | 扩散模型外部条件中的目标 σ；本次为 {args.target_conductivity:g} mS/cm。它不是实验实测值，而是生成方向。 |",
        f"| anion_ratio_condition | 扩散模型外部条件中的阴离子比例；本次固定 {condition:g}。 |",
        "| σ MAE / RMSE / MAPE | 全体可信配方的 σ 相对目标的绝对误差、均方根误差和 MAE/目标×100%。MAPE 是**相对指定目标**而非对实验真值的预测误差。 |",
        "| r MAE / RMSE | 全体可信配方的预测 anion ratio 相对输入条件的误差；r 指 anion ratio。 |",
        f"| Joint normalized MAE / RMSE | 先将每条配方的 σ 误差除以 {JOINT_SIGMA_SCALE:g} mS/cm、r 误差除以 {JOINT_ANION_RATIO_SCALE:g}，再对两个无量纲误差等权聚合。MAE 是两任务归一化绝对误差的平均；RMSE 是两任务归一化平方误差的均方根。数值越小越好。 |",
        "| σ ±1 / ±2 / ±5 | 预测 σ 落在目标 ±1、±2、±5 mS/cm 窗口内的比例。 |",
        "| r ±0.05 / ±0.10 | 预测 r 落在条件 r ±0.05、±0.10 窗口内的比例。 |",
        "| joint ±2/±0.05、±2/±0.10 | 同时满足 σ 和 r 两个窗口的比例，是联合条件生成的主覆盖指标。 |",
        "| 前 100 个联合排序候选 | 以 candidate_rank_score 排序后的同等 100 个候选。该表检查有限实验预算下的效果；不等同于独立测试集。 |",
        "| conductivity_pred_ms_cm / anion_ratio_pred | 对解码后配方复评分得到的 σ / r，用于报告内的条件误差与密度图。 |",
        "| temperature_c / salt_molar_ratio | 送入 oracle 复评分的温度与盐浓度条件；本次统一固定为 25 °C 与 0.10，故不是本轮生成器主动探索的变量。 |",
        "| bow | decoder 输出的候选词表组成向量；前段为溶剂比例、后段为盐比例。它用于还原具体配方与统计组成多样性。 |",
        "| candidate_rank_score | 候选排序分数；当前主要来自 oracle 联合目标评分，用于选出有限实验预算下的前 100 个，不能视为独立实验标签。 |",
        f"| prediction_status | `trusted` 表示 σ 有效且预测 anion ratio 位于 [{ratio_lower:g}, {ratio_upper:g}]；其余为 `out_of_domain`，保留在原始生成数中但不参与密度与命中统计。 |",
        "| explanation_residual_25c | 当前解释头在 25 °C 给出的 σ 残差项；本报告未以其作为 G0–G2 主排序标准，只保留用于后续机制复核。 |",
        "| n_sampled / n_trusted / n_out_of_domain | 总生成数 / 同时通过 σ 数值检查和 anion-ratio 窗口的可信数 / 其余候选。域外数高意味着候选更可能超出联合设计约束。 |",
        "| unique_compositions | 按活性溶剂与比例去重后的配方数，衡量生成多样性而非重复抽样。 |",
        "",
        "![Conductivity coverage](conductivity_coverage_heatmap.png)", "",
        "![Joint target-window coverage](joint_target_window_metrics_2x2.png)", "",
        "![All-formula separate and joint target-condition errors](target_joint_error_metrics_3x2.png)", "",
        "![Top-100 separate and joint target-condition errors](selected_joint_error_metrics.png)", "",
        "![Formula range](generated_formula_range.png)", "",
        "二维分面图按 **上排 G0 / G2-b、下排 G1 / G2** 排布；σ 覆盖图保留紧凑热图，行顺序同样为 G0、G2-b、G1、G2。`trusted-formula coverage` 的分母是该组通过基本有效性检查的可信生成配方数。`joint` 表示同一条配方同时落入 σ 与 r 两个指定窗口。不再生成单独的 target-hit-rate 或运行时间图。",
        "",
        "## σ—阴离子比例二维密度", "",
        f"横轴为冻结 oracle 预测 σ，纵轴为 oracle 预测阴离子比例；红虚线是 {args.target_conductivity:g} mS/cm，白点虚线是输入条件 {condition:g}。"
        "色条表示每个二维 bin 占全部 1024 个生成配方的比例，故域外样本不会因重新归一化而被隐藏。",
        "",
    ]
    if has_density_comparison:
        report.extend(["![2×2 density comparison](conductivity_anion_density_comparison.png)", ""])
    report.extend([
        "## 目标附近的具体配方", "",
        f"每组先按组成去重，再取预测 σ 最接近 {args.target_conductivity:g} mS/cm 的 10 个配方。G0/G2-b 使用 Bamboo 词表名称；其余组使用 SMILES，括号内为该溶剂在配方中的摩尔比例。"
        "图采用单行换行标签和加宽左边距，避免不同配方文字重叠。",
        "",
    ])
    for group, csv_name, figure in top_figures:
        report.extend([f"### {group}", f"![{group} top-10]({figure})", f"配方明细：[CSV]({csv_name})", ""])
    report.extend([
        "## 解释边界", "",
        "解释约束组的筛选来自片段归因和分子物性软约束，而非直接用测试集 σ 标签选分子。G1/G2-b/G2 的复评分来源为当前双任务 oracle；G0 记录的 `oracle_source` 为发布 Bamboo predictor。因此跨 G0 与其余组的数值差异只能作描述性比较，不应据此宣布模型优劣。所有“接近目标”指标均是 oracle 内部一致性，不是实验准确度；最终结论仍需独立 predictor 或实验验证。",
    ])
    report_text = "\n".join(report).replace("kappa_", "σ_").replace("κ", "σ")
    (root / "generation_experiment_report_zh.md").write_text(report_text + "\n", encoding="utf-8")
    print(f"Wrote joint-condition summary, 2x2 figures, and report to {root}")


if __name__ == "__main__":
    main()
