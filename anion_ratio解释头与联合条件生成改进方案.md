# Anion-ratio 解释头与联合条件生成改进方案

## 目标

在不牺牲 Bamboo 主干电导率预测精度的前提下，使模型能够：

1. 同时按目标电导率和目标阴离子比例生成配方；
2. 解释哪些溶剂片段、盐/阴离子片段、物性和交互作用导致阴离子比例变化；
3. 用联合命中率筛选可实验验证的候选。

## 已实施：联合条件 generator 数据

旧实现把所有合成配方都标记为固定 `anion_ratio=0.5`，即使 oracle 对不同配方给出了不同的预测比例，diffusion 也无法学习比例条件。

新实现对每个随机配方由冻结 oracle 输出：

```text
condition = [conductivity_oracle, anion_ratio_oracle]
```

并在“物理电导率 × anion ratio”的二维分箱中轮询抽样，减少训练样本集中于常见比例区域的偏差。推理阶段可输入：

```text
[20 mS/cm, 0.3] / [20 mS/cm, 0.5] / [20 mS/cm, 0.7]
```

注意：`anion_ratio_condition` 是 diffusion 的输入目标；`anion_ratio_pred` 是对生成配方再由 oracle 得到的输出。二者接近才说明条件控制有效。

## 后续解释头：双任务、但不改主预测路径

保持当前部署的 Bamboo 配方预测主干冻结，新增并行的 `anion_ratio_explain` 任务；不要把解释残差直接加回主 `readout_anion_ratio`。

```text
冻结的主预测器
├── conductivity 主输出
├── anion_ratio 主输出
├── conductivity_explain（现有）
└── anion_ratio_explain（新增）
```

这样解释分支的失败不会降低主电导率或主比例预测性能。

## anion-ratio 解释的输入与形式

仅用溶剂片段和 `dc / vis / Tm` 不足以解释 anion ratio。新增任务至少应包括：

- 溶剂片段主贡献；
- 盐/阴离子片段主贡献；
- 溶剂—盐交互；
- 盐浓度、温度条件基线；
- 必要时加入溶剂—溶剂—盐的低秩三元残差。

推荐在 logit 空间做保持加和一致的归因：

```text
logit(r_hat)
= base(concentration, temperature)
+ solvent_fragment_main
+ salt_anion_main
+ solvent_salt_interaction
+ ternary_residual

r_hat = sigmoid(logit(r_hat))
```

其中每一个加和项都可汇总回“片段—承载分子—盐种—条件”，而 sigmoid 保证输出位于 0–1。

## 训练目标

先用主模型对实验/MD 数据完成预测训练；随后冻结主干，仅训练解释头：

```text
L_anion_explain
= L(actual_anion_ratio, anion_explain_pred)
+ λ_fidelity · L(stopgrad(anion_main_pred), anion_explain_pred)
+ λ_main · L1(main_effect)
+ λ_pair · L1(interaction)
+ λ_residual · L2(residual)
```

`fidelity` 约束表示解释头应忠实复现主预测器的趋势；其权重应从小值开始调节，避免像电导率解释头那样压制解释能力。应同时报告：

- 主 anion-ratio 预测 MAE/R²；
- 解释头预测 MAE/R²；
- 主预测与解释预测的 fidelity MAE；
- 片段/盐贡献在 train、valid、test 中的符号稳定性。

## 联合生成评价

单独达到电导率目标不足以说明生成成功。对每个目标 `(sigma_target, r_target)` 报告：

```text
conductivity_error = |sigma_pred - sigma_target|
anion_error        = |r_pred - r_target|
joint_hit          = conductivity_error <= Delta_sigma
                     and anion_error <= Delta_r
```

可先采用 `Delta_sigma = 2 mS/cm`、`Delta_r = 0.05`，再按实验误差调整。G0–G3 应在相同的 `(sigma, r)` 网格、候选数和 oracle 下比较：

- 条件命中率；
- 联合命中率；
- Pareto 前沿（高电导率、低偏差、低黏度/高稳定性）；
- 温度—浓度鲁棒性。

## 推荐实施顺序

1. 用当前联合条件数据重训 G1/G2 diffusion 与 decoder；G3复用新的 G2 generator。
2. 在 `r_target = 0.3 / 0.5 / 0.7` 下进行条件采样并画 `anion_ratio_pred`—`conductivity_pred` 分布。
3. 报告 joint hit，而不仅是 conductivity hit。
4. 若联合条件控制仍差，再新增独立的 `anion_ratio_explain` 头。
5. 最后引入盐/阴离子片段和溶剂—盐交互，形成可用于生成约束的解释报告。
