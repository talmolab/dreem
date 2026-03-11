"""Module containing input/output data structures for easy storage and manipulation."""

__all__ = [
    "AssociationMatrix",
    "Config",
    "Frame",
    "FrameFlagCode",
    "Instance",
    "Track",
    "is_pretrained_shortname",
    "list_pretrained_models",
    "resolve_checkpoint",
    "resolve_config",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AssociationMatrix": ("dreem.io.association_matrix", "AssociationMatrix"),
    "Config": ("dreem.io.config", "Config"),
    "FrameFlagCode": ("dreem.io.flags", "FrameFlagCode"),
    "Frame": ("dreem.io.frame", "Frame"),
    "Instance": ("dreem.io.instance", "Instance"),
    "Track": ("dreem.io.track", "Track"),
    "is_pretrained_shortname": ("dreem.io.pretrained", "is_pretrained_shortname"),
    "list_pretrained_models": ("dreem.io.pretrained", "list_pretrained_models"),
    "resolve_checkpoint": ("dreem.io.pretrained", "resolve_checkpoint"),
    "resolve_config": ("dreem.io.pretrained", "resolve_config"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dreem.io' has no attribute {name!r}")
