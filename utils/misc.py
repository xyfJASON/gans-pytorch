"""Miscellaneous utility functions."""

import datetime
import numpy as np
import random
import torch
import sys


def check_freq(freq: int, step: int):
    """Check if the current step (0-indexed) is a multiple of the frequency."""
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    """Get the current time as a string in the format of 'YYYY-mm-dd-HH-MM-SS'."""
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def amortize(N: int, n: int):
    """Amortize N into several parts, each of which is n (except the last one).

    Args:
        N: The number to amortize.
        n: The amortization unit.

    Returns:
        A list of amortized parts. The last part may be less than n.
    """
    k, r = N // n, N % n
    return k * [n] if r == 0 else k * [n] + [r]


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def query_yes_no(question: str, default: str = "yes"):
    """Ask a yes/no question.

    Args:
        question: The question to ask.
        default: The default answer if the user just hits <Enter>.
         It must be "yes" (the default), "no" or None (meaning an answer is required).

    Returns:
        True if the answer is "yes" or False if the answer is "no".

    References:
      - https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
