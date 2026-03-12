"""Tracking Inference using GTR Model."""

__all__ = [
    "Tracker",
    "run_tracking",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Tracker": ("dreem.inference.tracker", "Tracker"),
    "run_tracking": ("dreem.inference.track", "run_tracking"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dreem.inference' has no attribute {name!r}")
