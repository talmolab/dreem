"""Top-level package for BioGTR."""

from biogtr.version import __version__

from .models.global_tracking_transformer import GlobalTrackingTransformer
from .models.gtr_runner import GTRRunner
from .models.transformer import Transformer
from .models.visual_encoder import VisualEncoder

from .io.frame import Frame
from .io.instance import Instance
from .io.association_matrix import AssociationMatrix
from .io.config import Config
from .io.visualize import annotate_video

# from .training import run

from .inference.tracker import Tracker
