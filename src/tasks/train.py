import torch
import os
import sys

from torch.utils.data import DataLoader

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

from src.configs import get_cfg_defaults
from src.AVL_config.config import shared_configs
from lib.data import make_data_loader
from lib.utils.config import init_config, Config
from lib.utils.logger import setup_logger_wandb as setup_logger
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.utils.miscellaneous import str_to_bool


def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument("--kinetics",
                        type=str,
                        default='400',
                        help="400 or 600")
    parser.add_argument("--att_mode",
                        type=str,
                        default='default',
                        help="default, full")
    parser.add_argument("--lambda_",
                        type=float,
                        default=0.5,
                        help="lambda_ loss")
    parser.add_argument("--pretrained_2d",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--freeze_passt',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--use_checkpoint',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--learn_mask_enabled',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument(
        '--transfer_method',
        type=int,
        default=0,
        help=
        "0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask"
    )
    parser.add_argument(
        '--att_mask_expansion',
        type=int,
        default=0,
        help=
        "-1: random init, 0: random init and then diag-based copy, 1: interpolation"
    )
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    args = base_config.parse_args()
    return args


def main(MAD_args, AVL_args):

    # Dataloader
    data_loader = make_data_loader(
        MAD_args,
        is_train=True,
        #is_distributed=distributed,
    )

    # Get Video Swin model
    swin_model = get_swin_model(AVL_args)
    # Get BERT and tokenizer
    bert_model, config, tokenizer = get_bert_model(AVL_args)
    # build SwinBERT based on training configs
    vl_transformer = VideoTransformer(AVL_args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=AVL_args.freeze_backbone)


if __name__ == '__main__':

    # Import arguments for AVL transformer:
    shared_configs.shared_video_captioning_config(cbs=True, scst= True)
    AVL_args = get_custom_args(shared_configs)
    
    # Import arguments for MAD dataloader:
    MAD_config_file = 'src/configs/mad.yaml'
    MAD_args = get_cfg_defaults()
    MAD_args.merge_from_file(MAD_config_file)

    # Set up logging:
    logger = setup_logger(init_config, MAD_args)

    logger.info("Loaded configuration file {}".format(MAD_config_file))
    with open(MAD_config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    
    logger.info("MAD dataset running with config:\n{}".format(MAD_args))
    logger.info("AVL transformer running with config:\n{}".format(AVL_args))
    
    main(MAD_args, AVL_args)