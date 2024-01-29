import torch
import os
import sys

from torch.utils.data import DataLoader

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

from src.configs import get_cfg_defaults
from lib.data import make_data_loader
from lib.utils.config import config, Config
from lib.utils.logger import setup_logger_wandb as setup_logger

        

def main():

    config_file = 'src/configs/mad.yaml'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)

    # Set up logging
    logger = setup_logger(config, cfg)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        #is_distributed=distributed,
    )

    print(data_loader)




if __name__ == '__main__':
    #args = load_config_to_args('src/configs/newconfig.yaml')
    main()