"""R0.5: Pickle-safe serializer for export artifact shard files.

The on-disk ``.pt`` shard files are standard PyTorch pickles.  Standard
``torch.load`` (without ``weights_only=True``) is vulnerable to arbitrary
code execution via malicious pickle data.  This module provides:

- ``SafeUnpickler`` — restricted unpickler allowing only torch tensors and
  Python primitives.  Rejects ``__reduce__``-based exploits.
- ``safe_torch_load`` — convenience wrapper that delegates to
  ``torch.load(weights_only=True)`` for production use.
"""

from __future__ import annotations

import builtins
import io
import pickle
from pathlib import Path
from typing import Any

import torch


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler — only known-safe types are allowed.

    Raises ``pickle.UnpicklingError`` if a dangerous or unrecognised class
    is encountered.
    """

    _SAFE_TORCH_MODULES: dict[str, frozenset[str]] = {
        "torch": frozenset({
            "Tensor", "Size", "dtype", "device",
        }),
        "torch._utils": frozenset({
            "_rebuild_tensor_v2", "_rebuild_tensor_v2_perspective",
            "_rebuild_tensor_v2_storage",
            "_rebuild_meta_tensor_no_storage",
            "_flatten_dense_tensors", "_unflatten_dense_tensors",
        }),
    }

    _SAFE_STORAGE_NAMES = frozenset({
        "FloatStorage", "HalfStorage", "BFloat16Storage",
        "DoubleStorage", "LongStorage", "IntStorage",
        "ShortStorage", "CharStorage", "ByteStorage", "BoolStorage",
        "ComplexFloatStorage", "ComplexDoubleStorage",
    })

    def find_class(self, module: str, name: str) -> type:
        safe_names = self._SAFE_TORCH_MODULES.get(module)
        if safe_names is not None and name in safe_names:
            cls = getattr(torch, name, None)
            if cls is not None:
                return cls
            cls = getattr(torch._utils, name, None)
            if cls is not None:
                return cls

        # Storage types can appear from various torch submodules
        if name in self._SAFE_STORAGE_NAMES:
            cls = getattr(torch, name, None)
            if cls is not None:
                return cls
            cls = getattr(torch._utils, name, None)
            if cls is not None:
                return cls

        if module == "torch.nn" and name == "Parameter":
            return torch.nn.Parameter

        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict

        if module == "builtins":
            safe_builtins = frozenset({
                "NoneType", "bool", "int", "float", "bytes", "str",
                "list", "tuple", "dict", "set", "frozenset",
            })
            if name in safe_builtins:
                return getattr(builtins, name)

        raise pickle.UnpicklingError(
            f"unsafe pickle type: module={module!r} name={name!r}"
        )


def safe_load(fp) -> Any:
    """Load a pickle using the restricted ``SafeUnpickler``.

    Args:
        fp: A binary file-like object (opened with ``"rb"``).

    Returns:
        The deserialised Python object.

    Raises:
        pickle.UnpicklingError: if the pickle contains unsafe types.
    """
    return SafeUnpickler(fp).load()


def safe_loads(data: bytes) -> Any:
    """Load a pickle from *bytes* using the safe unpickler."""
    return safe_load(io.BytesIO(data))


def safe_torch_load(path: str | Path) -> Any:
    """Load a ``.pt`` shard file using PyTorch's built-in ``weights_only=True``.

    This is the production path for loading export shards.  It delegates to
    ``torch.load(weights_only=True)`` which safely handles tensor storage
    pickles without allowing arbitrary code execution.

    Args:
        path: Path to the ``.pt`` file.

    Returns:
        The deserialised object.
    """
    return torch.load(path, weights_only=True)


__all__ = [
    "SafeUnpickler",
    "safe_load",
    "safe_loads",
    "safe_torch_load",
]
