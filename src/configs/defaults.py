import os
import yaml
import random

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "VLG"
_C.MODEL.PRETRAINV = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.STRIDE = 1
_C.INPUT.NUM_PRE_CLIPS   = 128
_C.INPUT.PRE_QUERY_SIZE  = 300
_C.INPUT.IMG_RES = 224
_C.INPUT.LANG_FEAT       = 'clip'  
_C.INPUT.NORMALIZE_VIDEO_FEATS= False
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for validation, as present in paths_catalog.py
_C.DATASETS.VAL = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.VLG = CN()
_C.MODEL.VLG.NUM_CLIPS = 128
_C.MODEL.VLG.NEG_PROB = 0.5

_C.MODEL.VLG.FEATPOOL = CN()
_C.MODEL.VLG.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.VLG.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.VLG.FEATPOOL.KERNEL_SIZE = 2
_C.MODEL.VLG.FEATPOOL.DROPOUT = 0.0
_C.MODEL.VLG.FEATPOOL.POS = 'none' # 'none', 'cos', 'learn'
_C.MODEL.VLG.FEATPOOL.GROUPS = 32
_C.MODEL.VLG.FEATPOOL.WIDTH_GROUP = 4
_C.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS = 1
_C.MODEL.VLG.FEATPOOL.NUM_NEIGHBOURS = 3 

_C.MODEL.VLG.FEAT2D = CN()
_C.MODEL.VLG.FEAT2D.POOLING_COUNTS = [15,8,8,8]

_C.MODEL.VLG.INTEGRATOR = CN()
_C.MODEL.VLG.INTEGRATOR.LSTM = CN()
_C.MODEL.VLG.INTEGRATOR.LSTM.NUM_LAYERS = 3
_C.MODEL.VLG.INTEGRATOR.LSTM.BIDIRECTIONAL = False
_C.MODEL.VLG.INTEGRATOR.LSTM.DROPOUT = 0.0
_C.MODEL.VLG.INTEGRATOR.DROPOUT_LINEAR = 0.0
_C.MODEL.VLG.INTEGRATOR.DROPOUT_SGCN = 0.0
_C.MODEL.VLG.INTEGRATOR.SKIP_CONN = True
_C.MODEL.VLG.INTEGRATOR.QUERY_HIDDEN_SIZE = 512
_C.MODEL.VLG.INTEGRATOR.NUM_AGGREGATOR_LAYERS = 1


_C.MODEL.VLG.MATCH = CN()
_C.MODEL.VLG.MATCH.ORDERING_EDGE = True
_C.MODEL.VLG.MATCH.SEMANTIC_EDGE = True
_C.MODEL.VLG.MATCH.MATCHING_EDGE = True
_C.MODEL.VLG.MATCH.NUM_NEIGHBOURS = 3 
_C.MODEL.VLG.MATCH.GROUPS = 32
_C.MODEL.VLG.MATCH.WIDTH_GROUP = 4
_C.MODEL.VLG.MATCH.DROPOUT_GM = 0.0

_C.MODEL.VLG.MOMENT_POOLING = CN()
_C.MODEL.VLG.MOMENT_POOLING.ATTENTION_MODE = 'cross_learnable'

_C.MODEL.VLG.PREDICTOR = CN() 
_C.MODEL.VLG.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.VLG.PREDICTOR.KERNEL_SIZE = 1
_C.MODEL.VLG.PREDICTOR.NUM_STACK_LAYERS = 8
_C.MODEL.VLG.PREDICTOR.POS =  'none' 
_C.MODEL.VLG.PREDICTOR.DROPOUT_CONV = 0.0

_C.MODEL.VLG.LOSS = CN()
_C.MODEL.VLG.LOSS.MIN_IOU = 0.3
_C.MODEL.VLG.LOSS.MAX_IOU = 0.7


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 50
_C.SOLVER.LR = 0.0001
_C.SOLVER.LR_STEP_SIZE = 10
_C.SOLVER.LR_GAMMA = 0.25
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.STRIDE = 32
_C.TEST.BATCH_SIZE = 1
_C.TEST.NMS_THRESH = 0.5
_C.TEST.STRIDE = 32


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

_C.rng_seed = 0
_C.load_path = None # none
_C.skip_test = False
_C.logname = None # none
_C.expname = None
_C.expid = None
_C.wandb = CN()
_C.wandb.use_wandb = False
_C.wandb.name = None
_C.wandb.tags = None
_C.wandb.project = 'FGmovieAD'  # name of the wandb project
_C.wandb.entity = 'pardoalejo'


# ---------------------------------------------------------------------------- #
# FAVD args
# ---------------------------------------------------------------------------- #
_C.img_res = 224
_C.vidswin_size = "base"
_C.kinetics = 600
_C.pretrained_2d = 0

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

    
def _select_value(v):
    if not isinstance(v, list):
        return v
    random.shuffle(v)
    return v[0]