"""Top-level package for BioGTR."""

from biogtr.version import __version__

from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models.gtr_runner import GTRRunner
from biogtr.models.transformer import Transformer
from biogtr.models.visual_encoder import VisualEncoder

from biogtr.io.frame import Frame
from biogtr.io.instance import Instance
from biogtr.io.association_matrix import AssociationMatrix
from biogtr.io.config import Config
from biogtr.io.visualize import annotate_video

# from .training import run

from biogtr.inference.tracker import Tracker
