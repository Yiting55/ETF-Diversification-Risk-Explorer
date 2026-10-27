"""Refresh engine modules and version UI caches when engine source changes."""
import hashlib
import importlib
from pathlib import Path
from threading import RLock


_LOCK = RLock()
# Dependencies first: imported functions/classes must all refer to the new modules.
_MODULES = ("metrics", "scenarios", "data", "correlation", "analyze", "whatif", "risk_contributions")


def load_engine():
    """Return the current engine package and its source revision.

    Streamlit hashes a cached wrapper, but not the imported functions it calls.
    A long-running server can also retain old module objects after a UI edit.
    Refresh once per source revision, before looking up any cached analysis.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (*_MODULES, "__init__"):
        digest.update((root / f"{name}.py").read_bytes())
    revision = digest.hexdigest()
    with _LOCK:
        package = importlib.import_module("engine")
        if getattr(package, "_source_revision", None) != revision:
            for name in _MODULES:
                importlib.reload(importlib.import_module(f"engine.{name}"))
            importlib.reload(package)
            package._source_revision = revision
    return package, revision
