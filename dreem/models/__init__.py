"""Model architectures and layers."""

__all__ = [
    "Embedding",
    "FourierPositionalEmbeddings",
    "GlobalTrackingTransformer",
    "GTRRunner",
    "Transformer",
    "DescriptorVisualEncoder",
    "VisualEncoder",
    "create_visual_encoder",
    "register_encoder",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Embedding": ("dreem.models.embedding", "Embedding"),
    "FourierPositionalEmbeddings": (
        "dreem.models.embedding",
        "FourierPositionalEmbeddings",
    ),
    "GlobalTrackingTransformer": (
        "dreem.models.global_tracking_transformer",
        "GlobalTrackingTransformer",
    ),
    "GTRRunner": ("dreem.models.gtr_runner", "GTRRunner"),
    "Transformer": ("dreem.models.transformer", "Transformer"),
    "DescriptorVisualEncoder": (
        "dreem.models.visual_encoder",
        "DescriptorVisualEncoder",
    ),
    "VisualEncoder": ("dreem.models.visual_encoder", "VisualEncoder"),
    "create_visual_encoder": ("dreem.models.visual_encoder", "create_visual_encoder"),
    "register_encoder": ("dreem.models.visual_encoder", "register_encoder"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dreem.models' has no attribute {name!r}")
