import os
import logging


class StatusTracker:
    """Track status and print to logger and tensorboard."""
    def __init__(
            self,
            logger: logging.Logger,
            print_freq: int,
            tensorboard_dir: str,
            is_main_process: bool = True,
    ):
        self.logger = logger
        self.print_freq = print_freq

        self.tb_writer = None
        if is_main_process:
            self.tb_writer = get_tb_writer(log_dir=tensorboard_dir)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()

    def track_status(self, name: str, status: dict, step: int):
        message = f'[{name}] step: {step}'
        for i, (k, v) in enumerate(status.items()):
            message += f', {k}: {v:.6f}'
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f'{name}/{k}', v, step)
        if self.print_freq > 0 and (step + 1) % self.print_freq == 0:
            self.logger.info(message)


def get_tb_writer(log_dir: str):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
    return tb_writer
