import os
import torch


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
