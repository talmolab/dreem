"""Top-level package for dreem."""

# from .training import run
from dreem.inference.tracker import Tracker
from dreem.io.association_matrix import AssociationMatrix
from dreem.io.config import Config
from dreem.io.frame import Frame
from dreem.io.instance import Instance
from dreem.io.visualize import annotate_video
from dreem.models.global_tracking_transformer import GlobalTrackingTransformer
from dreem.models.gtr_runner import GTRRunner
from dreem.models.transformer import Transformer
from dreem.models.visual_encoder import VisualEncoder
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
