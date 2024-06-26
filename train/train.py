# -*- coding: utf-8 -*-
"""cifar10 autoattack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hARkBQpRwNh-iFNBJKLOqKNQEcKk8Qgm
"""

# !pip install git+https://github.com/fra31/auto-attack

# https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
# Load in relevant libraries, and alias where appropriate
# from autoattack import AutoAttack 
import argparse
# import os
import sys
# from tqdm import tqdm
# import numpy as np
import torch
import wandb
import yaml
from distributed_train import Trainer, load_train_objects, set_seeds

def parse_args(args: list) -> argparse.Namespace:
  """Parse command line parameters.

  :param args: command line parameters as list of strings (for example
      ``["--help"]``).
  :return: command line parameters namespace.
  """
  parser = argparse.ArgumentParser(
      description="Train the models for this experiment."
  )

  parser.add_argument(
      "--no-cuda", action="store_true", default=False, help="disables CUDA training"
  )
  parser.add_argument(
      "--dataset-path",
      default="/home-local2/jongn2.extra.nobkp/data",
      help="the path to the dataset",
      type=str,
  )
  parser.add_argument(
      "--cpus-per-trial",
      default=1,
      help="the number of CPU cores to use per trial",
      type=int,
  )
  parser.add_argument(
      "--project-name",
      help="the name of the Weights and Biases project to save the results",
      # required=True,
      default='',
      type=str,
  )
  parser.add_argument(
      "--dataset",
      help="dataset used",
      required=False,
      type=str,
  )
  parser.add_argument(
      "--loss",
      help="loss used",
      required=False,
      type=str,
  )
  parser.add_argument(
      "--paramfile",
      help="param file to use",
      default="",
      required=False,
      type=str,
    )

  return parser.parse_args(args)


def run_experiment(params: dict, args: argparse.Namespace) -> None:
    train_objects = load_train_objects(params, args, distributed=False)
    torch.autograd.set_detect_anomaly(True)
    mode = 'online' if params['wandblog'] else 'disabled'
    wandb.init(project="refactoredAT", name=args.project_name, mode=mode)
    modeltrainer = Trainer(train_objects, params)
    modeltrainer.train()
    wandb.finish()
    torch.autograd.set_detect_anomaly(False)

def main(args: list) -> None:
  """Parse command line args, load training params, and initiate training.

  :param args: command line parameters as list of strings.
  """
  args = parse_args(args)
  paramsfilename = f'./params{args.paramfile}.yaml'
  print(paramsfilename)
  with open(paramsfilename, 'r') as param_file:
      params = yaml.load(param_file, Loader=yaml.SafeLoader)

  if args.loss:
      params['loss'] = args.loss
  if args.dataset:
      params['dataset'] = args.dataset 

  set_seeds(params['seed'])
  run_experiment(params, args)


if __name__ == "__main__":
  main(sys.argv[1:])