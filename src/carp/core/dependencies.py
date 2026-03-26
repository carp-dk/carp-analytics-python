"""Optional dependency helpers."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def module_available(module_name: str) -> bool:
    """Return whether a module can be imported."""

    return importlib.util.find_spec(module_name) is not None


def import_or_raise(module_name: str, extra_name: str) -> Any:
    """Import a dependency or raise a helpful runtime error.

    Args:
        module_name: Importable module name.
        extra_name: Package extra or install hint shown to the user.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the dependency is unavailable.
    """

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised through callers.
        raise RuntimeError(f"{module_name} is required for this feature. Install the `{extra_name}` extras.") from exc
