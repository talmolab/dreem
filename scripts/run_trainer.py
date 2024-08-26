from dreem.training import train
from omegaconf import OmegaConf
import os

# /Users/mustafashaikh/dreem/dreem/training
# /Users/main/Documents/GitHub/dreem/dreem/training
os.chdir("/Users/main/Documents/GitHub/dreem/dreem/training")
base_config = "./configs/base.yaml"
# params_config = "./configs/override.yaml"

cfg = OmegaConf.load(base_config)
# cfg["params_config"] = params_config

train.run(cfg)