"""Top-level package for dreem."""

from dreem.version import __version__

__all__ = [
    "Tracker",
    "AssociationMatrix",
    "Config",
    "Frame",
    "Instance",
    "annotate_video",
    "GlobalTrackingTransformer",
    "GTRRunner",
    "Transformer",
    "VisualEncoder",
    "__version__",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Tracker": ("dreem.inference.tracker", "Tracker"),
    "AssociationMatrix": ("dreem.io.association_matrix", "AssociationMatrix"),
    "Config": ("dreem.io.config", "Config"),
    "Frame": ("dreem.io.frame", "Frame"),
    "Instance": ("dreem.io.instance", "Instance"),
    "annotate_video": ("dreem.io.visualize", "annotate_video"),
    "GlobalTrackingTransformer": (
        "dreem.models.global_tracking_transformer",
        "GlobalTrackingTransformer",
    ),
    "GTRRunner": ("dreem.models.gtr_runner", "GTRRunner"),
    "Transformer": ("dreem.models.transformer", "Transformer"),
    "VisualEncoder": ("dreem.models.visual_encoder", "VisualEncoder"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module 'dreem' has no attribute {name!r}")


def setup_logging():
    """Setup logging based on `logging.yaml`."""
    import logging
    import logging.config
    import os

    import yaml

    package_directory = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(package_directory, "..", "logging.yaml"), "r") as stream:
        logging_cfg = yaml.load(stream, Loader=yaml.FullLoader)

    logging.config.dictConfig(logging_cfg)
