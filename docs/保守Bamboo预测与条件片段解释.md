# 保守 Bamboo 预测与条件片段解释

此模式不让片段头修改 Bamboo FT 的预测。对每个配方，部署用的输出始终是：

```
conductivity_pred = conductivity_base
```

因此，只要加载的 `base_formula_ft.pt` 是原版 Bamboo 的 FT checkpoint，此 pipeline 的预测精度与其一致。

同时，固定的 FragmentBlock 为每个分子提供片段表示。独立解释头在 FT 实验数据上学习：

```
conductivity_explain = 条件基准(salt, 浓度, 温度)
                     + 片段主贡献
                     + 跨分子片段对交互贡献
```

主贡献先按同一分子的片段数归一化，交互只保留不同分子的片段对，总片段效应使用有界残差保证稳定。每个样本导出的主贡献和片段对之和与上式严格一致。

启动：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 bash submit/run_fragment_explanation_pipeline.sh all \
  > run/fragment_explanation_pipeline.out 2>&1 &
```

结果中同时保留 `conductivity_pred` 和 `conductivity_explain_pred`。前者用于主模型精度，后者用于评估归因头是否能重现实验趋势；不应把它当作替代主模型的预测值。
