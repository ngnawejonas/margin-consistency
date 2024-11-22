# Margin Consistency
Paper: ["Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers."](https://arxiv.org/abs/2406.18451)

## Installation

1. Clone this repository with `git clone git@github.com:ngnawejonas/margin-consistency.git`
2. Install the requirements `pip install -r requirements.txt`

## Running the code

1. Edit the configuration file `params.yaml` to specify:
   * the attack (fab for fab attack, cw for carlini-wagner, clever for clever score or otherwise for auto-attack) 
   * a folder to save the results
2. Run `$python3 eval.py` or use the script `run.sh` 

## Analysis

Use `notebook/xpLinf-TrainDNet.ipynb` for analysis

## Citation

If you used this code, please cite our paper:

```
@inproceedings{
ngnawe2024detecting,
title={Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers},
author={Jonas Ngnawe and Sabyasachi Sahoo and Yann Batiste Pequignot and Frederic Precioso and Christian Gagn{\'e}},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=XHCYZNmqnv}
}
```
