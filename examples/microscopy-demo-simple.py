# /// script
# requires-python = "==3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pathlib",
#     "matplotlib",
#     "os",
#     "omegaconf",
#     "dreem",
#     "torch",
#     "huggingface_hub",
# ]
# ///

import os
import torch
import numpy as np
from omegaconf import OmegaConf
from dreem.inference import track
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download