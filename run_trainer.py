from dreem.training import train
from omegaconf import OmegaConf
import os

os.chdir("./dreem/training")
base_config = "./configs/base.yaml"
# params_config = "/path/to/override.yaml"

cfg = OmegaConf.load(base_config)
# cfg["params_config"] = params_config

train.run(cfg)