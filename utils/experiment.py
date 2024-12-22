"""Utilities for managing experiments."""

import os
import sys
import tqdm
import shutil
import importlib
from typing import List, Dict, Union
from omegaconf import OmegaConf, DictConfig

from .misc import get_time_str, query_yes_no


def create_exp_dir(
        exp_dir: str,
        conf_yaml: str,
        subdirs: List[str] = ('ckpt', ),
        time_str: str = None,
        exist_ok: bool = False,
        cover_dir: bool = False,
):
    """Create the experiment directory.

    Args:
        exp_dir: The path to the experiment directory.
        conf_yaml: A string of the configuration in YAML format.
        subdirs: The subdirectories to create in the experiment directory.
        time_str: The time string to append to the configuration file name.
        exist_ok: Whether to allow the directory to exist. Note that some files may be overwritten if True.
        cover_dir: Whether to cover the directory if it already exists. Note that all files will be removed if True.
    """
    # Check if the directory exists
    if os.path.exists(exp_dir) and not exist_ok:
        cover = cover_dir or query_yes_no(
            question=f'{exp_dir} already exists! Cover it anyway?',
            default='no',
        )
        shutil.rmtree(exp_dir, ignore_errors=True) if cover else sys.exit(1)

    # Make directories
    os.makedirs(exp_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    # Write configuration
    if time_str is None:
        time_str = get_time_str()
    with open(os.path.join(exp_dir, f'config-{time_str}.yaml'), 'w') as f:
        f.write(conf_yaml)


def instantiate_from_config(conf: Union[Dict, DictConfig], **extra_params):
    """Instantiate an object from a configuration dictionary.

    The configuration dictionary should follow the format:
    ---------------------------------
    target: 'module.submodule.Class'
    params:
        param1: value1
        param2: value2
    ---------------------------------
    An object will be instantiated as `module.submodule.Class(param1=value1, param2=value2, **extra_params)`.

    Args:
        conf: The configuration dictionary.
        extra_params: Extra parameters to pass to the class constructor.

    Returns:
        The instantiated object.
    """
    if isinstance(conf, DictConfig):
        conf = OmegaConf.to_container(conf)
    module, cls = conf['target'].rsplit('.', 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    params = conf.get('params', dict())
    params.update(extra_params)
    return cls(**params)


def find_resume_checkpoint(exp_dir: str, resume: str):
    """Find the checkpoint directory to resume training.

    Checkpoints are assumed to be saved in the 'ckpt' subdirectory.
    A typical checkpoint directory structure is:
    ---------------------------------
    exp_dir/
    ├── ckpt/
    │   ├── best/
    │   │   ├── model.pt
    │   │   └── optimizer.pt
    │   ├── step0009999/
    │   │   ├── model.pt
    │   │   └── optimizer.pt
    │   ├── ...
    │   └── step0999999/
    │       ├── model.pt
    │       └── optimizer.pt
    └── ...
    ---------------------------------

    Args:
        exp_dir: The path to the experiment directory.
        resume: The checkpoint to resume training. It can be 'best', 'latest', or a specific checkpoint directory.

    Returns:
        The path to the checkpoint directory.
    """
    if os.path.isdir(resume):
        ckpt_path = resume
    elif resume == 'best':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'best')
    elif resume == 'latest':
        d = dict()
        for name in os.listdir(os.path.join(exp_dir, 'ckpt')):
            if os.path.isdir(os.path.join(exp_dir, 'ckpt', name)) and name[:4] == 'step':
                d.update({int(name[4:]): name})
        ckpt_path = os.path.join(exp_dir, 'ckpt', d[sorted(d)[-1]])
    else:
        raise ValueError(f'resume option {resume} is invalid')
    assert os.path.isdir(ckpt_path), f'{ckpt_path} is not a directory'
    return ckpt_path


def discard_label(x):
    return x[0] if isinstance(x, (list, tuple)) else x
