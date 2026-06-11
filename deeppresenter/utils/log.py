import functools
import inspect
import logging
import time
import traceback
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from deeppresenter.utils.constants import LOGGING_LEVEL

_context_logger: ContextVar[logging.Logger | None] = ContextVar(
    "_context_logger", default=None
)


def create_logger(name: str = __name__, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOGGING_LEVEL)
    fmt = logging.Formatter(
        "%(levelname)-4s %(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_logger(name: str = __name__, log_file: str | Path | None = None) -> logging.Logger:
    logger = create_logger(name, log_file)
    _context_logger.set(logger)
    return logger


def get_logger() -> logging.Logger:
    ctx = _context_logger.get()
    if ctx is None:
        ctx = create_logger("deeppresenter.default")
        _context_logger.set(ctx)
    return ctx


def debug(msg, *args, **kwargs): get_logger().debug(msg, *args, **kwargs)
def info(msg, *args, **kwargs): get_logger().info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs): get_logger().warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs): get_logger().error(msg, *args, **kwargs)


class timer:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        if elapsed > 1:
            debug(f"{self.name} took {elapsed:.2f}s")

    def __call__(self, func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    if elapsed > 1:
                        debug(f"{self.name or func.__name__} took {elapsed:.2f}s")
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    if elapsed > 1:
                        debug(f"{self.name or func.__name__} took {elapsed:.2f}s")
            return sync_wrapper


def logging_openai_exceptions(identifier: Any, exc: Exception) -> str:
    msg = f"Exception [{type(exc).__name__}]: {str(exc)}\n{traceback.format_exc()}"
    warning(f"{identifier} → {msg}")
    return msg
