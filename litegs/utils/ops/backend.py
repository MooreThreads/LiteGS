from contextlib import contextmanager
from enum import Enum


class Backend(str, Enum):
    AUTO = "auto"
    SCRIPT = "script"
    CUDA = "cuda"


_DEFAULT_BACKEND = Backend.AUTO


def normalize_backend(backend: Backend | str | None) -> Backend:
    if backend is None:
        return _DEFAULT_BACKEND
    return Backend(backend)


def get_default_backend() -> Backend:
    return _DEFAULT_BACKEND


def set_default_backend(backend: Backend | str) -> None:
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = Backend(backend)


@contextmanager
def use_backend(backend: Backend | str):
    previous = get_default_backend()
    set_default_backend(backend)
    try:
        yield
    finally:
        set_default_backend(previous)
