# Electrolyte Mechanistic Explainer

这是从 `ele_predictor_clean` 提取的精简独立仓库，用于在不改变 Bamboo-Mixer 最终电导率预测的前提下，训练和分析条件感知的“性质—片段”解释头。

## 保留的功能

```text
冻结 Bamboo FT 主预测
        ↓
conductivity_pred（最终性能筛选）

固定 FragmentBlock + 冻结分子多任务性质预测
        ↓
条件基准（盐种、浓度、温度）
+ dc / vis / Tm 溶剂性质贡献
+ 溶剂片段主贡献
+ 跨溶剂片段对交互
        ↓
conductivity_explain_pred（实验趋势与结构/性质解释）
```

其中性质、片段及片段对均为模型条件关联，不能直接视为独立化学因果效应。

## 不复制的内容

本仓库不包含 checkpoint、训练结果、日志、运行数据、旧贡献残差实验、数据预处理脚本和生成器实现。已有预处理数据可在本仓库建立 `run/data/` 链接后直接复用。

## 运行

准备好已有的预处理数据后，运行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
  bash submit/run_property_fragment_explanation_pipeline.sh all \
  > run/property_fragment_explanation_pipeline.out 2>&1 &
```

脚本自动从既有 Bamboo-Mixer checkpoint 与旧 Fragment sidecar checkpoint 构建：

```text
run/checkpoints/fixed_fragment_mono.pt
run/checkpoints/base_formula_ft.pt
```

不会修改这些源 checkpoint。

可分步执行：

```bash
bash submit/run_property_fragment_explanation_pipeline.sh train
bash submit/run_property_fragment_explanation_pipeline.sh predict
bash submit/run_property_fragment_explanation_pipeline.sh explain
```

## 主要输出

```text
run/property_fragment_explanation/results/ft_test_predictions.json
run/property_fragment_explanation_artifacts/properties/
run/property_fragment_attribution/
```

每个配方 JSON 同时包括：

- `conductivity_pred`：冻结 Bamboo 主预测；
- `conductivity_explain_pred`：条件基准、性质项与片段项的可加和解释预测；
- `property_explanations`：`dc`、`vis`、`Tm` 的配方级和逐溶剂性质贡献；
- `fragment_explanations`：片段主贡献及跨溶剂片段对交互。

详见 [电导率配方解释方案.md](电导率配方解释方案.md)。
