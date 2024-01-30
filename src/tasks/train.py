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
from lib.utils.config import init_config, Config
from lib.utils.logger import setup_logger_wandb as setup_logger

from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer



def main():

    config_file = 'src/configs/mad.yaml'
    args = get_cfg_defaults()
    args.merge_from_file(config_file)

    # Set up logging
    logger = setup_logger(init_config, args)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(args))

    #data_loader = make_data_loader(
    #    args,
    #    is_train=True,
        #is_distributed=distributed,
    #)

    # Get Video Swin model
    swin_model = get_swin_model(args)
    # Get BERT and tokenizer
    bert_model, config, tokenizer = get_bert_model(args)
    # build SwinBERT based on training configs
    vl_transformer = VideoTransformer(args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)


if __name__ == '__main__':
    #args = load_config_to_args('src/configs/newconfig.yaml')
    main()