# Code for "Highlighting Object Category Immunity for the Generalization of Human-Object Interaction Detection"

Xinpeng Liu*, Yong-lu Li*, Cewu Lu, accepted to AAAI-2022
Under construction

## Train

### Separate training for different sterams:

```python
python train.py --config_path configs/human.yml --exp human
python train.py --config_path configs/spatial.yml --exp spatial
python train.py --config_path configs/object.yml --exp object
```

### Calibration aware unified inference:

```python
python train.py --config_path configs/unified.yml --exp unified
```

## Calculation of mPD

Please refer to `HICO_DET_utils.py` for details.

## TODO List

- [ ] Data preparation script
- [ ] Trained parameters
- [ ] Code cleaning
