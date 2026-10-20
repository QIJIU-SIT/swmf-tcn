"""Safe, deterministic loading boundary for official external model trees."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType


class ExternalSourceCollision(RuntimeError):
    """More than one external model family was requested in one process."""


_active_family: str | None = None
_active_root: Path | None = None


def _module_is_from_root(module: ModuleType, root: Path) -> bool:
    """Return whether an imported module has files below ``root``."""
    locations: list[str] = []
    origin = getattr(module, "__file__", None)
    if origin:
        locations.append(str(origin))
    module_path = getattr(module, "__path__", ())
    if isinstance(module_path, (str, bytes)):
        locations.append(str(module_path))
    else:
        try:
            locations.extend(str(location) for location in module_path or ())
        except TypeError:
            # Some extension modules expose a non-iterable namespace object.
            # Their ``__file__`` (if any) was already checked above.
            pass
    for location in locations:
        try:
            Path(location).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        else:
            return True
    return False


def _purge_modules_from_root(root: Path) -> None:
    """Remove only modules whose files belong to one external source tree."""
    for name, module in tuple(sys.modules.items()):
        if module is not None and _module_is_from_root(module, root):
            del sys.modules[name]


def source_tree_sha256(root: Path, *, patterns: tuple[str, ...] = ("**/*.py",)) -> str:
    root = Path(root).resolve()
    files = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    if not files:
        raise FileNotFoundError(f"no source files found under {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_external_module(
    source_root: Path, *, family: str, module: str, required_files: tuple[str, ...] = (),
) -> ModuleType:
    global _active_family, _active_root
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"external source root not found: {root}")
    missing = [relative for relative in required_files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("external source is missing: " + ", ".join(missing))
    if _active_family is not None and _active_family != family:
        raise ExternalSourceCollision("only one external model family may be loaded per process")
    if _active_root is not None and _active_root != root:
        _purge_modules_from_root(_active_root)
        sys.path[:] = [item for item in sys.path if item != str(_active_root)]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _active_family, _active_root = str(family), root
    return importlib.import_module(module)


def reset_external_source_state() -> None:
    """Reset loader state for isolated test processes only."""
    global _active_family, _active_root
    if _active_root is not None:
        _purge_modules_from_root(_active_root)
        sys.path[:] = [item for item in sys.path if item != str(_active_root)]
    _active_family, _active_root = None, None
