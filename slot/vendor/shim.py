"""Minimal stand-ins for the three helpers slot/vendor needs from videosaur.utils.

videosaur.utils imports pytorch_lightning/torchmetrics, which we do not want; only these
three pure-python helpers are used by the vendored modules.
"""
import functools
import sys
from typing import Any, Callable, Dict, Iterable, Optional


def config_as_kwargs(config, to_filter=None, defaults=None) -> Dict[str, Any]:
    always_filter = ("name",)
    flt = tuple(always_filter) + tuple(to_filter) if to_filter else always_filter
    items = list((defaults or {}).items()) + list(config.items())
    return {k: v for k, v in items if k not in flt}


def get_class_by_name(module_name: str, name: str):
    pymodule = sys.modules[module_name]
    return pymodule.__dict__.get(name, None)


def make_build_fn(pymodule_name: str, group_name: str):
    def decorator(wrapped_fn: Callable):
        @functools.wraps(wrapped_fn)
        def build_fn(config, name=None, default_name=None, **kwargs):
            if config is None:
                raise ValueError(f"No config specified while building {group_name}")
            name = name or config.get("name") or default_name
            if name is None:
                raise ValueError(f"No name specified while building {group_name}")
            module = wrapped_fn(config, name, **kwargs)
            if module is None:
                cls = get_class_by_name(pymodule_name, name)
                if cls is not None:
                    module = cls(**config_as_kwargs(config), **kwargs)
            return module
        return build_fn
    return decorator
