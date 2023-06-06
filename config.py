import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'pcqm4m'
# Dataset source
_C.DATA.SOURCE = 'ogb'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 16
# Max nodes
_C.DATA.TRAIN_MAX_NODES = 512
_C.DATA.INFER_MAX_NODES = 512
# Multi hop max distance
_C.DATA.MULTI_HOP_MAX_DIST = 20
# spatial pos max
_C.DATA.SPATIAL_POS_MAX = 1024
# number of atoms
_C.DATA.NUM_ATOMS = 4608
# numebr of edges
_C.DATA.NUM_EDGES = 1536
# number of in degree
_C.DATA.NUM_IN_DEGREE = 512
# number of out degree
_C.DATA.NUM_OUT_DEGREE = 512
# number of edge distance
_C.DATA.NUM_EDGE_DIST = 128
# number of spatial
_C.DATA.NUM_SPATIAL = 512
# edge type
_C.DATA.EDGE_TYPE = 'multi_hop'
# metric name
_C.DATA.METRIC = 'MAE'
# task type
_C.DATA.TASK_TYPE = 'graph_regression'
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'GPTrans'
# Model name
_C.MODEL.NAME = 'GPTrans'
# Pretrained weight from checkpoint, could be pcqm4m pretrained weight
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.0
# Drop path type
_C.MODEL.DROP_PATH_TYPE = 'linear'  # linear, uniform
# Attention drop rate
_C.MODEL.ATTN_DROP_RATE = 0.0

# GPTRANS parameters
_C.MODEL.GPTRANS = CN()
_C.MODEL.GPTRANS.NUM_LAYERS = 24
_C.MODEL.GPTRANS.NUM_HEADS = 23
_C.MODEL.GPTRANS.NODE_DIM = 736
_C.MODEL.GPTRANS.EDGE_DIM = 92
_C.MODEL.GPTRANS.LAYER_SCALE = None
_C.MODEL.GPTRANS.MLP_RATIO = 1.0
_C.MODEL.GPTRANS.POST_NORM = False
_C.MODEL.GPTRANS.NUM_CLASSES = 1

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = None
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# criterion type
_C.TRAIN.CRITERION = 'mse'
# class weights
_C.TRAIN.CLASS_WEIGHTS = None
# reduce zero label
_C.TRAIN.REDUCE_ZERO_LABEL = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# ZeRO
_C.TRAIN.OPTIMIZER.USE_ZERO = False

# EMA
_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998

# FLAG
# self.flag_m = cfg.flag_m
# self.flag_step_size = cfg.flag_step_size
# self.flag_mag = cfg.flag_mag
_C.TRAIN.FLAG = CN()
_C.TRAIN.FLAG.ENABLE = False
_C.TRAIN.FLAG.FLAG_M = 3
_C.TRAIN.FLAG.FLAG_STEP_SIZE = 0.01
_C.TRAIN.FLAG.FLAG_MAG = 0

# LR_LAYER_DECAY
_C.TRAIN.LR_LAYER_DECAY = False
_C.TRAIN.LR_LAYER_DECAY_RATIO = 0.875

# FT head init weights
_C.TRAIN.RAND_INIT_FT_HEAD = False
_C.TRAIN.PARAM_EFFICIENT_TUNING = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.RANDOM_FEATURE = False


# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# eval freq
_C.EVAL_FREQ = 1
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

_C.AMP_TYPE = 'float16'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'dataset') and args.dataset:
        config.DATA.DATASET = args.dataset
    if hasattr(args, 'source') and args.source:
        config.DATA.SOURCE = args.source
    if hasattr(args, 'data_path') and args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if hasattr(args, 'pretrained') and args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
    if hasattr(args, 'accumulation_steps') and args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if hasattr(args, 'amp_opt_level') and args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if hasattr(args, 'output') and args.output:
        config.OUTPUT = args.output
    if hasattr(args, 'tag') and args.tag:
        config.TAG = args.tag
    if hasattr(args, 'eval') and args.eval:
        config.EVAL_MODE = True
    if hasattr(args, 'throughput') and args.throughput:
        config.THROUGHPUT_MODE = True
    if hasattr(args, 'save_ckpt_num') and args.save_ckpt_num:
        config.SAVE_CKPT_NUM = args.save_ckpt_num
    if hasattr(args, 'use_zero') and args.use_zero:
        config.TRAIN.OPTIMIZER.USE_ZERO = True
    # set local rank for distributed training
    if hasattr(args, 'local_rank') and args.local_rank:
        config.LOCAL_RANK = args.local_rank

    # output folder
    config.MODEL.NAME = args.cfg.split('/')[-1].replace('.yaml', '')
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
