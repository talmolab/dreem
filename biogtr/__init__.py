"""Top-level package for BioGTR."""

from biogtr.version import __version__
from biogtr.models.attention_head import MLP, ATTWeightHead
from biogtr.models.feature_encoders.visual_encoder import VisualEncoder
from biogtr.models.transformer import Transformer
