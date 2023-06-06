import os
import math
import torch
import numpy as np
import torch.distributed as dist
from collections import OrderedDict
from timm.utils import get_state_dict
from sklearn.metrics import confusion_matrix
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_ema_checkpoint(config, model_ema, logger):
    logger.info(
        f'==============> Resuming form {config.MODEL.RESUME}....................'
    )
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME,
                                                        map_location='cpu',
                                                        check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    assert isinstance(checkpoint, dict)
    if 'model_ema' in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_ema'].items():
            if model_ema.ema_has_module:
                name = 'module.' + k if not k.startswith('module') else k
            else:
                name = k
            new_state_dict[name] = v
        msg = model_ema.ema.load_state_dict(new_state_dict, strict=False)
        logger.info(msg)
        logger.info('Loaded state_dict_ema')
    else:
        logger.warning(
            'Failed to find state_dict_ema, starting from loaded model weights'
        )

    best_performance_ema = 0
    if 'best_performance_ema' in checkpoint:
        best_performance_ema = checkpoint['best_performance_ema']
    if 'ema_decay' in checkpoint:
        model_ema.decay = checkpoint['ema_decay']
    return best_performance_ema


def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    logger.info(
        f'==============> Resuming form {config.MODEL.RESUME}....................'
    )
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME,
                                                        map_location='cpu',
                                                        check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    print('resuming model')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    best_performance = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if optimizer is not None:
            print('resuming optimizer')
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('resume optimizer failed')
        if lr_scheduler is not None:
            print('resuming lr_scheduler')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != 'O0' and checkpoint[
                'config'].AMP_OPT_LEVEL != 'O0':
            scaler.load_state_dict(checkpoint['amp'])
        logger.info(
            f"=> loaded successfully {config.MODEL.RESUME} (epoch {checkpoint['epoch']})"
        )
        if 'best_performance' in checkpoint:
            best_performance = checkpoint['best_performance']

    del checkpoint
    torch.cuda.empty_cache()

    return best_performance


def load_pretrained(config, model, logger):
    logger.info(
        f'==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......'
    )
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    if "ema" in config.MODEL.PRETRAINED:
        state_dict = checkpoint['model_ema']
        logger.info(f'==============> Loading ema weight......')
    else:
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']

    if config.TRAIN.RAND_INIT_FT_HEAD:
        model.head.weight.data = model.head.weight.data * 0.001
        model.head.bias.data = model.head.bias.data * 0.001
        del state_dict['head.weight']
        del state_dict['head.bias']
        logger.warning(f'Re-init classifier head to 0!')
        
    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)
    logger.info(f'=> loaded successfully {config.MODEL.PRETRAINED}')

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, scaler, logger, best_performance,
                    model_ema=None, ema_decay=None, best_performance_ema=None, best=None):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'config': config
    }
    if model_ema is not None:
        save_state['model_ema'] = get_state_dict(model_ema)
    if ema_decay is not None:
        save_state['ema_decay'] = ema_decay
    if best_performance is not None:
        save_state['best_performance'] = best_performance
    if best_performance_ema is not None:
        save_state['best_performance_ema'] = best_performance_ema
    if config.AMP_OPT_LEVEL != 'O0':
        # save_state['amp'] = amp.state_dict()
        save_state['amp'] = scaler.state_dict()
    if best is None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{best}.pth')
    logger.info(f'{save_path} saving......')
    torch.save(save_state, save_path)
    logger.info(f'{save_path} saved !!!')

    if dist.get_rank() == 0 and isinstance(epoch, int):
        to_del = epoch - config.SAVE_CKPT_NUM * config.SAVE_FREQ
        old_ckpt = os.path.join(config.OUTPUT, f'ckpt_epoch_{to_del}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f'All checkpoints founded in {output_dir}: {checkpoints}')
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints],
            key=os.path.getmtime)
        print(f'The latest checkpoint founded: {latest_checkpoint}')
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def all_gather_tensor(tensor_list, tensor):
    dist.all_gather(tensor_list, tensor)
    return tensor_list


class ApexScalerWithGradNormCount:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None,
                 create_graph=False, update_grad=True):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip_grad)
            optimizer.step()
    
    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


# https://github.com/facebookresearch/ConvNeXt/blob/main/utils.py
class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class MyAverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, max_len=-1):
        self.val_list = []
        self.count = []
        self.max_len = max_len
        self.val = 0
        self.avg = 0
        self.var = 0

    def update(self, val):
        self.val = val
        self.avg = 0
        self.var = 0
        if not math.isnan(val) and not math.isinf(val):
            self.val_list.append(val)
        if self.max_len > 0 and len(self.val_list) > self.max_len:
            self.val_list = self.val_list[-self.max_len:]
        if len(self.val_list) > 0:
            self.avg = np.mean(np.array(self.val_list))
            self.var = np.std(np.array(self.val_list))


def accuracy_SBM(scores, targets):
    # build confusion matrix
    S = targets.cpu().numpy()
    if scores.size(-1) == 1:
        C = torch.sigmoid(scores) < 0.5
        C = (torch.where(C, 0, 1)).cpu().numpy()
        C = C.squeeze(-1)
    else:
        C = scores.argmax(-1).cpu().numpy()
    CM = confusion_matrix(S, C).astype(np.float32)
    # calculate accuracy
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r] / float(cluster.shape[0])
            if CM[r,r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    print("pre classes acc:", pr_classes)
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return torch.tensor(acc, device=scores.device)
