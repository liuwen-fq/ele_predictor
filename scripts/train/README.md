# Training
## Configuration file
- configuration files are stored as yaml files under [config/train](../../config/train/) 


## Predictive model training 
```
python3 scripts/train/train.py \
  --conf config/train/predictor_config.yaml 
```

## Fragment-aware molecular pretraining

Prepare the fragment dataset first, set `meta.work_folder` in the training
configuration, then run:

```
python3 scripts/train/train.py \
  --conf config/train/mono_fragment_config.yaml \
  --task mono
```
