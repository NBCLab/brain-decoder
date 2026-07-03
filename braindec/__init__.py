"""Braindec: Brain image decoder."""

from importlib import import_module

__all__ = [
    "dataset",
    "embedding",
    "fetcher",
    "loss",
    "model",
    "plot",
    "train",
    "utils",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
