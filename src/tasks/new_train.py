import torch
import yaml
import argparse
import os
import sys
import json

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

from torch.utils.data import DataLoader
from src.datasets.dataloader import create_dataloader


def to_namespace(config):
    """ Convert a nested dictionary to nested Namespaces. """
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = to_namespace(value)
        return argparse.Namespace(**config)
    else:
        return config

def load_config_to_args(file_path='src/configs/newconfig.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return to_namespace(config)

        

def main(args):

    dataloader = create_dataloader(args)


if __name__ == '__main__':
    args = load_config_to_args('src/configs/newconfig.yaml')
    main(args)