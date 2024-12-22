import os

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    if is_dist_avail_and_initialized():
        return int(os.environ['LOCAL_RANK'])
    return 0


def is_main_process():
    return get_rank() == 0


def on_main_process(function):
    def wrapper(*args, **kwargs):
        if is_main_process():
            return function(*args, **kwargs)
    return wrapper


def wait_for_everyone():
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def init_distributed_mode():
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl')
        dist.barrier()
    else:
        device = torch.device('cuda')
    return device


def reduce_tensor(tensor):
    if is_dist_avail_and_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= get_world_size()
        return rt
    return tensor


def gather_tensor(tensor):
    if is_dist_avail_and_initialized():
        tensor_list = [torch.ones_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    return [tensor]
