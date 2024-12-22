import tqdm
import logging


def get_logger(
        name: str = 'exp',
        log_level: int = logging.INFO,
        log_file: str = None,
        file_mode: str = 'w',
        use_tqdm_handler: bool = False,
        is_main_process: bool = True,
):
    """Get a logger that prints to both console and file.

    Args:
        name: The name of the logger.
        log_level: The logging level. Note that levels of non-main processes are always 'ERROR'.
        log_file: The path to the log file. If None, the log file is disabled.
        file_mode: The mode to open the log file.
        use_tqdm_handler: Whether to use TqdmLoggingHandler.
        is_main_process: Whether the logger is for the main process.
    """
    logger = logging.getLogger(name)
    # Check if the logger exists
    if logger.hasHandlers():
        return logger
    # Add a stream handler
    if not use_tqdm_handler:
        stream_handler = logging.StreamHandler()
    else:
        stream_handler = TqdmLoggingHandler()
    handlers = [stream_handler]
    # Add a file handler for main process
    if is_main_process and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    # Set format & level for all handlers
    # Note that levels of non-main processes are always 'ERROR'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_level = log_level if is_main_process else logging.ERROR
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that uses tqdm.write() to print log messages.

    This handler prevents the logging messages from interfering with tqdm progress bars. Note that
    you need to use `import tqdm` instead of `from tqdm import tqdm` to make this handler work.

    References:
      - https://stackoverflow.com/a/38739634/23025233
    """
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)
