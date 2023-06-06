from torch import optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from apex.optimizers import FusedAdam
import logging
logger = logging.getLogger(__name__)


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = set_weight_decay_and_lr(
        model,
        config.TRAIN.WEIGHT_DECAY,
        config.TRAIN.BASE_LR,
        skip,
        skip_keywords,
        lr_layer_decay=config.TRAIN.LR_LAYER_DECAY,
        lr_layer_decay_ratio=config.TRAIN.LR_LAYER_DECAY_RATIO,
    )

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    use_zero = config.TRAIN.OPTIMIZER.USE_ZERO
    if use_zero:
        logger.info(f"\nUse Zero!")
        if opt_lower == 'sgd':
            # an ugly implementation
            # https://github.com/pytorch/pytorch/issues/71347
            optimizer = ZeroRedundancyOptimizer(
                parameters[0]['params'],
                optimizer_class=optim.SGD,
                momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                nesterov=True,
                lr=config.TRAIN.BASE_LR,
                weight_decay=config.TRAIN.WEIGHT_DECAY)
            if len(parameters[1]['params']) > 0:
                optimizer.add_param_group({
                    "params": parameters[1]['params'],
                    'weight_decay': 0.
                })
        elif opt_lower == 'adamw':
            optimizer = ZeroRedundancyOptimizer(
                parameters[0]['params'],
                optimizer_class=optim.AdamW,
                eps=config.TRAIN.OPTIMIZER.EPS,
                betas=config.TRAIN.OPTIMIZER.BETAS,
                lr=config.TRAIN.BASE_LR,
                weight_decay=config.TRAIN.WEIGHT_DECAY)
            if len(parameters[1]['params']) > 0:
                optimizer.add_param_group({
                    "params": parameters[1]['params'],
                    'weight_decay': 0.
                })
    else:
        if opt_lower == 'sgd':
            optimizer = optim.SGD(parameters,
                                  momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                  nesterov=True,
                                  lr=config.TRAIN.BASE_LR,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY)
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    eps=config.TRAIN.OPTIMIZER.EPS,
                                    betas=config.TRAIN.OPTIMIZER.BETAS,
                                    lr=config.TRAIN.BASE_LR,
                                    weight_decay=config.TRAIN.WEIGHT_DECAY)
        elif opt_lower == 'fused_adamw':
            optimizer = FusedAdam(parameters,
                                  eps=config.TRAIN.OPTIMIZER.EPS,
                                  betas=config.TRAIN.OPTIMIZER.BETAS,
                                  lr=config.TRAIN.BASE_LR,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY)
    return optimizer


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_keywords_in_dict(name, keywords_dict):
    for k, v in keywords_dict.items():
        if k in name:
            return v
    return None


def set_weight_decay_and_lr(model, weight_decay, base_lr, skip_list=(), skip_keywords=(),
                            lr_layer_decay=None, lr_layer_decay_ratio=None, layerwise_lr=True):
    parameters = {}
    no_decay_name = []
    lr_ratio_log = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # 1. check wd
        if len(param.shape) == 1 or name.endswith(".bias") or (
                name in skip_list) or check_keywords_in_name(
                    name, skip_keywords):
            wd = 0.
            no_decay_name.append(name)
        else:
            wd = weight_decay

        # 2. set weight_decay
        if lr_layer_decay:
            logger.info('layer-wise lr decay is used !')
            assert hasattr(model, 'lr_decay_keywords')
            lr_ratio_keywords = model.lr_decay_keywords(lr_layer_decay_ratio)
            # 2. check lr
            ratio = check_keywords_in_dict(name, lr_ratio_keywords)
            if ratio is not None:
                lr = ratio * base_lr
            else:
                lr = base_lr

            lr_ratio_log[name] = (base_lr, ratio, wd, param.requires_grad)
        else:
            lr = base_lr
        group_name = f"weight_decay_{str(wd)}_lr_{str(lr)}"
        if group_name not in parameters:
            parameters[group_name] = {'params': [param], 'weight_decay': wd, 'lr': lr}
        else:
            parameters[group_name]['params'].append(param)

    logger.info(f'no decay params: {no_decay_name}')
    if layerwise_lr:
        logger.info('lr_ratio_params:')
        for k, v in lr_ratio_log.items():
            print(k, v)
    parameters = list(parameters.values())
    return parameters
