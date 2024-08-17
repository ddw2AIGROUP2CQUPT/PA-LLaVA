from typing import Tuple, Union
import torch
import torch.nn.functional as F
# from .p2i_ops import p2i
import math
from torch import nn


def resize_embedding(embedding_layer, new_size, num_tokens=1, mode='bicubic'):
    """Resize the position embedding in an nn.Embedding layer.

    Args:
        embedding_layer (nn.Embedding): The embedding layer to resize.
        new_size (int): The new size for the positional embedding.
        num_tokens (int): The number of special tokens (e.g., CLS token).
        mode (str): The interpolation mode.

    Returns:
        nn.Embedding: A new embedding layer with resized position embedding.
    """
    # Extract weights from the original embedding layer
    original_weights = embedding_layer.weight.data
    
    # Resize the weights using the provided function
    resized_weights = _resize_pe(original_weights, new_size, mode, num_tokens)
    
    # Create a new embedding layer and initialize it with the resized weights
    new_embedding_layer = nn.Embedding(resized_weights.size(0), resized_weights.size(1))
    new_embedding_layer.weight.data = resized_weights
    
    return new_embedding_layer


def _resize_pe(pe: torch.Tensor, new_size: int, mode: str = 'bicubic', num_tokens: int = 1) -> torch.Tensor:
    """Resize positional embeddings.

    Args: 
        pe (torch.Tensor): A tensor with shape (num_tokens + old_size ** 2, width). pe[0, :] is the CLS token.

    Returns:
        torch.Tensor: A tensor with shape (num_tokens + new_size **2, width).
    """
    l, w = pe.shape
    old_size = int(math.sqrt(l-num_tokens))
    assert old_size ** 2 + num_tokens == l
    return torch.cat([
        pe[:num_tokens, :],
        F.interpolate(pe[num_tokens:, :].reshape(1, old_size, old_size, w).permute(0, 3, 1, 2),
                      (new_size, new_size), mode=mode, align_corners=False).view(w, -1).t()], dim=0)


def normalize_points(points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ Normalize coordinates to [0, 1].
    """
    return (points + 0.5) / torch.tensor([[[w, h]]]).to(points)

def denormalize_points(normalized_points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ Reverse normalize_points.
    """
    return normalized_points * torch.tensor([[[w, h]]]).to(normalized_points) - 0.5

# def points2heatmap(normalized_points, heatmap_size: Tuple[int, int], kernel_radius: float):
#     """ Normalized points [b x npoints x 2(XY)] -> heatmaps.
#     """
#     batch, npoints, _ = normalized_points.shape
#     out_h, out_w = heatmap_size

#     points = denormalize_points(normalized_points, out_h, out_w)

#     # (batch x npoints) x 1 x h x w
#     heatmap = torch.zeros(
#         batch * npoints, 1, out_h, out_w).to(points)
#     # (batch x npoints) x 2
#     points_flatten = points.view(-1, 2)
#     # (batch x npoints)
#     batch_inds = torch.arange(
#         batch * npoints, dtype=torch.int32).cuda()
#     # (batch x npoints) x 1
#     points_color = torch.ones(
#         points_flatten.size(0), 1).to(points_flatten)
#     # (batch x npoints) x 1 x h x w
#     heatmap = p2i(points_flatten, points_color, batch_inds=batch_inds, background=heatmap,
#                   kernel_radius=kernel_radius,
#                   kernel_kind_str='gaussian_awing', reduce='max')
#     # batch x npoints x h x w
#     heatmap = heatmap.reshape(batch, npoints, out_h, out_w)
#     return heatmap

def heatmap2points(heatmap, t_scale: Union[None, float, torch.Tensor] = None):
    """ Heatmaps -> normalized points [b x npoints x 2(XY)].
    """
    dtype = heatmap.dtype
    _, _, h, w = heatmap.shape

    # 0 ~ h-1, 0 ~ w-1
    yy, xx = torch.meshgrid(
        torch.arange(h).float(),
        torch.arange(w).float())

    yy = yy.view(1, 1, h, w).to(heatmap)
    xx = xx.view(1, 1, h, w).to(heatmap)

    if t_scale is not None:
        heatmap = (heatmap * t_scale).exp()
    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

    yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # b x npoints
    xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # b x npoints

    points = torch.stack([xx_coord, yy_coord], dim=-1)  # b x npoints x 2

    normalized_points = normalize_points(points, h, w)
    return normalized_points


def _expand_as_rgbs(x):
    _, c, _, _ = x.shape
    if c == 3:
        return [x]

    if c % 3 > 0:
        x = torch.cat([
            x, x[:, [-1], :, :].expand(
                -1, 3 - c % 3, -1, -1)], dim=1)
    c = x.size(1)
    assert c % 3 == 0
    return list(x.split([3] * (c // 3), dim=1))


def _visualize_flags(flags, size, num_flags):
    batch_size = flags.size(0)
    flags = flags.to(dtype=torch.uint8)
    has_what = [flags & torch.full_like(flags, 1 << i)
                for i in range(num_flags)]
    # batch x 1 x 1 x 4
    vis_im = torch.stack(has_what, dim=1).float().view(
        batch_size, 1, 1, num_flags)
    vis_im = F.interpolate(vis_im.expand(-1, 3, -1, -1),
                           size=size, mode='nearest')
    return vis_im


# def visualize_in_row(*data) -> torch.Tensor:
#     """Visualize data in one row.

#     Args:
#         *data (list): A list of (value, modal, [v_min, v_max]) tuples.

#         Each tuple defines the following inputs:

#             value (torch.Tensor): The data value to visualize.
#             modal (str): The modal type string of the data.
#                 Supported data modal types are:

#                 * "BHW", "BNHW", "BHWN" for tensors;
#                 * "flags_{K}" for binary flags, with K being the number of bits;
#                 * "points" for points, where `value` is a tensor with shape [B, N, 2].

#             v_min (float): Optional, to normalize value.
#             v_max (float): Optional, to normalize value.

#     Returns:
#         torch.Tensor: A tensor with shape b x 3 x h x w.
#     """
#     batch = None
#     size = None
#     device = None

#     row = []
#     for v in data:
#         assert isinstance(v, (tuple, list))
#         if len(v) == 2:
#             value, modal = v
#             v_min, v_max = 0.0, 1.0
#         elif len(v) == 4:
#             value, modal, v_min, v_max = v
#         else:
#             raise RuntimeError(
#                 'Input either (value, modal) or (value, modal, v_min, v_max)')

#         if value is None:
#             assert batch is not None
#             assert size is not None
#             assert device is not None
#             value = torch.rand(batch, 1, size[0], size[1], device=device)
#             modal = 'BNHW'
#             v_min, v_max = 0.0, 1.0

#         if modal == 'BHW':
#             assert isinstance(value, torch.Tensor)
#             value = value.detach().float()

#             batch = value.size(0)
#             size = value.shape[1:]
#             device = value.device

#             value = (value - v_min) / (v_max - v_min)
#             row.append(value.unsqueeze(
#                 1).expand(-1, 3, -1, -1))

#         elif modal == 'BNHW':
#             assert isinstance(value, torch.Tensor)
#             value = value.detach().float()

#             batch = value.size(0)
#             size = value.shape[2:]
#             device = value.device

#             value = (value - v_min) / (v_max - v_min)
#             row += _expand_as_rgbs(value)

#         elif modal == 'BHWN':
#             assert isinstance(value, torch.Tensor)
#             value = value.detach().float().permute(0, 3, 1, 2)

#             batch = value.size(0)
#             size = value.shape[2:]
#             device = value.device

#             value = (value - v_min) / (v_max - v_min)
#             row += _expand_as_rgbs(value)

#         elif modal.startswith('flags_'):
#             assert isinstance(value, torch.Tensor)
#             value = value.detach().float()

#             batch = value.size(0)
#             device = value.device

#             num_flags = int(modal.split('_')[1])
#             assert size is not None
#             row.append(_visualize_flags(value, size, num_flags))

#         elif modal == 'points':
#             points, background = value

#             if background is None:
#                 background = torch.rand(
#                     batch, 1, size[0], size[1], device=device)
#             else:
#                 assert isinstance(background, torch.Tensor)
#                 background = background.detach().float()
#                 background = (background - v_min) / (v_max - v_min)

#             if points is None:
#                 canvas = background
#             else:
#                 assert isinstance(points, torch.Tensor)
#                 points = points.detach().float()
#                 points = denormalize_points(
#                     points, background.size(2), background.size(3))

#                 npoints = points.size(1)
#                 batch = background.size(0)
#                 assert points.size(0) == batch
#                 channels = background.size(1)

#                 points = points.reshape(npoints * batch, 2)

#                 point_colors = torch.ones(
#                     npoints * batch, channels, dtype=background.dtype, device=background.device)
#                 batch_inds = torch.arange(batch).unsqueeze(1).expand(-1, npoints).reshape(
#                     npoints * batch).to(dtype=torch.int32, device=background.device)
#                 canvas = p2i(points, point_colors, batch_inds, background, 5)

#             row.append(canvas)

#     return torch.cat(row, dim=-1)


import math
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)        
        
        