"""Model registry: architecture entry points registered by name, as in timm."""
import fnmatch

_model_entrypoints = {}
_model_to_module = {}

__all__ = ["register_model", "list_models", "is_model", "model_entrypoint"]


def register_model(fn):
    """Register a function that builds a model architecture under the function's name."""
    name = fn.__name__
    if name in _model_entrypoints:
        raise ValueError("Duplicate model name: " + name)
    _model_entrypoints[name] = fn
    _model_to_module[name] = fn.__module__.split(".")[-1]
    return fn


def list_models(filter=""):
    """Sorted names of the registered architectures, optionally filtered by an fnmatch pattern."""
    names = sorted(_model_entrypoints)
    if filter:
        names = [n for n in names if fnmatch.fnmatch(n, filter)]
    return names


def is_model(name):
    return name in _model_entrypoints


def model_entrypoint(name):
    if name not in _model_entrypoints:
        raise RuntimeError("Unknown model (%s); registered: %s" % (name, ", ".join(list_models())))
    return _model_entrypoints[name]
