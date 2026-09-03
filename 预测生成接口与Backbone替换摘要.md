# 预测—生成接口与可替换 Backbone 摘要

## 目标

将当前“预测模型—synthetic 数据构建—条件生成—配方复评分”流程整理为可替换预测 backbone 的模块化接口。未来可比较不同预测模型、接入 RDF/MD 相互作用表征，或在客户数据上微调，而不需要重写生成器主体。

## 建议的统一接口

对任一预测 backbone 增加 `PredictorAdapter`，使其对输入配方提供统一输出：

```text
formula → PredictorAdapter → {
    sigma_pred,          # 普通 σ 预测
    anion_ratio_pred,    # 普通 r 预测
    formula_emb,         # 配方级连续表征
    mechanism_emb,       # 可选机制表征
    metadata             # 单位、标签变换、模型与数据版本
}
```

其中：

- `sigma_pred`、`anion_ratio_pred` 用于 frozen-oracle synthetic 标签、生成配方复评分和候选排序；
- `formula_emb` 作为 diffusion 学习与 decoder 重建的公式级表征；
- `mechanism_emb` 当前可为解释头交互代理 `[Iσ, Ir]`，未来可替换为 RDF、局部配位、溶剂化结构或其他 MD 表征；
- `metadata` 记录 σ 的变换方式、r 的尺度、嵌入维度、checkpoint、训练数据版本与机制表征定义，保证结果可追溯。

整体数据流为：

```text
配方 → PredictorAdapter → {σ̂, r̂, formula_emb, mechanism_emb}
                         ↓
                SyntheticDataBuilder
                         ↓
             条件 Diffusion + Formula Decoder
                         ↓
                配方生成与独立复评分
```

## 当前模型的对应关系

当前普通预测头为：

```text
conductivity  → σ̂
anion_ratio   → r̂
```

当前解释头提供：

```text
conductivity_explain.interaction_sum → Iσ
anion_ratio_explain.interaction_sum  → Ir
```

G2-M 的 diffusion 条件扩展为：

\[
[\sigma,\ r,\ I_\sigma,\ I_r]
\]

生成时并不使用生成后配方的解释结果作为条件；而是从训练集学习的经验先验

\[
p(I_\sigma,I_r\mid\sigma,r)
\]

中采样机制状态，再送入 diffusion。这避免了“先生成、再用同一配方解释结果回填条件”的循环使用。

## 需要解除的当前耦合

1. **公式嵌入维度**

   当前生成器默认使用 384 维 `formula_emb`。若新 backbone 输出维度为 \(d\)，增加：

   ```text
   EmbeddingAdapter: R^d → R^384
   ```

   即可保持 diffusion 与 decoder 结构不变；也可将生成器的嵌入维度改为配置项。

2. **预测器加载方式**

   当前 synthetic 标注与复评分直接依赖 `Mono + MolMix`。后续应由 `PredictorAdapter` 统一封装模型配置、checkpoint、输入图构建、标签逆变换和 explanation 输出。

3. **机制表征语义**

   `mechanism_emb` 必须随 metadata 一并保存，明确其来源、尺度、标准化方式和维度。当前 `[Iσ, Ir]` 是模型内部解释代理，不应被表述为物理相互作用能；未来 RDF embedding 则可成为具有明确 MD 定义的机制输入。

## 替换 Backbone 的实验原则

- 对每个 backbone，使用其自身的 `PredictorAdapter` 构建 synthetic 标签和机制表征；
- 保持词表、配方采样先验、训练规模、扩散架构、随机种子和生成预算一致；
- 若比较“不同 backbone 哪个更准确”，不能只用各自的 oracle 自评。应增加固定实验测试集、外部模拟/实验标签，或独立 evaluator；
- 若检验“机制条件是否有用”，使用同一 backbone 下的 Q / M / M-shuffle 对照：
  - Q：仅条件 \((σ,r)\)；
  - M：真实配方—机制配对 \((σ,r,Iσ,Ir)\)；
  - M-shuffle：在相同 \((σ,r,\text{溶剂数})\) 格内打乱完整机制对。

## 面向 RDF 的演进路径

```text
当前阶段： [Iσ, Ir] 解释头交互代理
      ↓
预训练阶段： RDF / 配位结构 / 分子间相互作用 embedding
      ↓
应用阶段： 用户性质数据微调 σ、溶解度、库伦效率等任务头
```

生成器只依赖标准化的 `mechanism_emb` 接口，因此能够平滑从当前解释头代理过渡到更物理的 RDF 监督表征。

## 结论

当前预测—生成流程具备模块化基础。补齐 `PredictorAdapter`、`EmbeddingAdapter` 与统一 metadata 规范后，可在不改变生成器核心逻辑的情况下替换预测 backbone，并将当前解释交互代理自然升级为未来的 RDF/MD 机制表征。
