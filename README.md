# Interpretable Electrolyte Property Prediction

This repository is an independent research extension of **Bamboo-Mixer** for
electrolyte formulation and conductivity prediction. It is developed on top
of the open-source work published by the ByteDance Seed team, while focusing
on extending the molecular representation and interpretability of the
predictive model.

## Acknowledgement

We sincerely thank the ByteDance Seed team for publishing **Bamboo-Mixer** and
for providing the original implementation and accompanying resources. Their
work established the three-stage predictive framework used as the foundation
of this repository:

1. Molecular-level pretraining with molecular property labels.
2. Formulation-level pretraining with molecular dynamics data.
3. Experimental fine-tuning with measured conductivity data.

The original paper and implementation are available at:

- Paper: [A Unified Predictive and Generative Solution for Liquid Electrolyte Formulation](https://arxiv.org/abs/2504.18728)
- Original repository: [ByteDance-Seed/bamboo_mixer](https://github.com/ByteDance-Seed/bamboo_mixer)
- Original data and checkpoints: [ByteDance-Seed/bamboo_mixer](https://huggingface.co/ByteDance-Seed/bamboo_mixer)

## Current Research Direction

The main extension in this repository is a chemistry-constrained **fragment
encoder** at the molecular level. Molecules are decomposed into chemically
meaningful fragments while preserving rings, functional groups, and the
connections between fragments. Fragment representations are then aggregated
inside each molecule and integrated with the original molecular graph
representation.

This design is intended to make electrolyte conductivity prediction more
interpretable, especially for understanding how solvent fragments and their
functional roles contribute to the final molecular and formulation-level
representation. The original Bamboo-Mixer prediction pipeline is retained as
the baseline so that the effect of fragment encoding can be evaluated under
the same data and training protocol.

## Training Framework

The predictive model follows the original three-stage organization:

1. **Molecular pretraining:** multi-task prediction of molecular properties
   with fragment-aware molecular encoding.
2. **MD formulation pretraining:** prediction of formulation properties from
   molecular dynamics data.
3. **Experimental fine-tuning:** fine-tuning on measured conductivity data and
   evaluating on an independent experimental test set.

The current implementation concentrates on molecular-level encoding. Parameter-
level attribution and intervention methods are intentionally reserved for a
later stage, after the fragment-aware model has been shown to preserve the
conductivity prediction accuracy of the original baseline.

Detailed instructions for preprocessing, training, and conductivity
evaluation are available in:

- [Data preparation instructions](./scripts/prepare_data/README.md)
- [Training instructions](./scripts/train/README.md)
- [Fragment model training and conductivity evaluation](./执行片段模型训练与电导率评估.md)
- [Inference instructions](./scripts/test_results/README.md)

## Molecular and Formulation Data

The repository uses two main data structures inherited from Bamboo-Mixer:

- `MonoData`: single-molecule structures and molecular property labels.
- `FormulaData`: solvent-salt formulations with concentration, temperature,
  conductivity, and anion-ratio information.

The original datasets and checkpoints can be obtained from the Bamboo-Mixer
resources linked above. New preprocessing outputs, fragment graphs, model
checkpoints, and evaluation results should be kept in local working directories
and should not be confused with the original published artifacts.

## Future Work

The next stage will extend the model toward **multi-task downstream transfer**.
Potential downstream tasks include conductivity, viscosity, density, ion
transport-related properties, and other experimentally measured electrolyte
properties. The goal is to reuse the fragment-aware molecular representation
across related tasks while retaining chemically meaningful fragment-level
interpretations.

## License

This project follows the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
of the original Bamboo-Mixer codebase. Please retain the original attribution
and citation when using or redistributing derivative work.

## Citation

If you use the original Bamboo-Mixer framework, please cite:

```bibtex
@misc{yang2025unifiedpredictivegenerativesolution,
      title={A Unified Predictive and Generative Solution for Liquid Electrolyte Formulation},
      author={Zhenze Yang and Yifan Wu and Xu Han and Ziqing Zhang and Haoen Lai and Zhenliang Mu and Tianze Zheng and Siyuan Liu and Zhichen Pu and Zhi Wang and Zhiao Yu and Sheng Gong and Wen Yan},
      year={2025},
      eprint={2504.18728},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2504.18728}
}
```
