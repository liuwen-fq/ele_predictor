# Data preparation

## Configuration file
- configuration files are stored as yaml files under [config/prepare_data](../../config/prepare_data/) 

## Single molecule inference dataset

`mono_config.yaml` declares `task: inference`; its records require one
measurement `temperature` field.
```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/mono_config.yaml \
  --data_type mono
```

## Fragment-aware molecular pretraining dataset

This configuration declares `task: pretrain`, reads `run/mono_data.json`, normalizes its molecular
properties, builds the atom graph and chemistry-constrained fragment graph, and
writes a separate processed dataset without changing the source JSON.

```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/mono_fragment_config.yaml \
  --data_type mono
```

## Electrolyte dataset
```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/formula_config.yaml \
  --data_type formula
```

## Train dataset
```
python3 scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/train_data_config.yaml \
  --data_type formula
```
