from dreem.training import train
from omegaconf import OmegaConf

base_config = "dreem/training/configs/base.yaml"
# params_config = "/path/to/override.yaml"

cfg = OmegaConf.load(base_config)
# cfg["params_config"] = params_config

train.run(cfg)