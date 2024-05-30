# BioGTR
Global Tracking Transformers for biological multi-object tracking.

## Installation
<!-- ### Basic
```
pip install biogtr
``` -->
### Development
1. Clone the repository:
```
git clone https://github.com/talmolab/biogtr && cd biogtr
```
2. Set up in a new conda environment:
```
conda env create -y -f environment.yml && conda activate biogtr
```

### Uninstalling
```
conda env remove -n biogtr
```

## Usage

Here we describe a basic workflow from setting up data thru training and running inference
### Background

This repo uses [`hydra`](https://hydra.cc) for config handling, and [`pytorch`](https://pytorch.org)/[`pytorch-lightning`](https://lightning.ai) to handle high-level training/eval/inference. Thus, we recommend skimming through their respective docs to get some familiarity but is not necessary.

## Step 1. Data Acquisition.

### Step 1.1: Get detections
One of the main advantages of this system is that we *decouple* detection and tracking. This means that our model assumes that you already have detections available when you are ready to track. In order to get these detections we recommend a couple methods. For animal pose-estimation, we highly recommend heading over to [SLEAP](https://sleap.ai) and running through their work flow. For microscopy tracking, check out [TrackMate](https://imagej.net/plugins/trackmate/) as well as [CellPose](https://www.cellpose.org) and [StarDist](https://imagej.net/plugins/stardist). If you are only doing tracking (ie inference), then you once you have your detections you are good to go! 
### ***FOR TRAINING*** Step 1.2 Proofreading
Otherwise, we recommend using the `sleap-label` gui to proofread your track labels. For animal tracking, if you used SLEAP you can start proofreading right away. Otherwise if you used a different system (e.g DeepLabCut) check out [`sleap.io.convert`](https://sleap.ai/api/sleap.io.convert.html#module-sleap.io.convert) for available converters. With microscopy, we highly recommend starting out with TrackMate and then proofreading in `sleap`. Here is [a converter from trackmate's output to a `.slp`](https://gist.github.com/aaprasad/5243be0785a40e9dafa1697ce2258e3e) file. In general, you can use [`sleap-io`](https://io.sleap.ai/latest/) to write a custom converter to `.slp` if you'd like to use the sleap-gui for proofreading
### Step 1.3 Organize data.

Although, our data loading does take paths to label files and video files directly so it's fairly flexible, we recommend organizing your data with a couple things in mind.

1. Match video and labels file stems. Because our datasets just take in a list of video files and a list of label files, the order in which the corresponding files are passed must match. (e.g `[file1.slp, file2.slp], [vid1.mp4, vid2.mp4]`). In order to make programmatic file searching easy its best to save your labels and vid files with the same stem so that you can ensure the ordering will be consistent. Its also just best practice so you know which video a labels file corresponds to.
2. Store corresponding videos and labels files in the same directory. This will again make searching for the files much easier. 

The best practice is to have your dataset organized as follows:
```bash
dataset_name/
    train/
        vid_1.{VID_EXTENSION}
        vid_1.{LABELS_EXTENSION}
        ...
        vid_n.{VID_EXTENSION}
        vid_n.{LABELS_EXTENSION}
    val/
        vid_1.{VID_EXTENSION}
        vid_1.{LABELS_EXTENSION}
        ...
        vid_n.{VID_EXTENSION}
        vid_n.{LABELS_EXTENSION}
    test/
        vid_1.{VID_EXTENSION}
        vid_1.{LABELS_EXTENSION}
        ...
        vid_n.{VID_EXTENSION}
        vid_n.{LABELS_EXTENSION}
    inference/
        vid_1.{VID_EXTENSION}
        vid_1.{LABELS_EXTENSION} # these files don't need tracks
        ...
        vid_n.{VID_EXTENSION}
        vid_n.{LABELS_EXTENSION}
```

## Step 2. Training

Now that you have your dataset set up, we can start training! In [`biogtr.training.train`](https://github.com/talmolab/biogtr/blob/main/biogtr/training/train.py), we provide a train script that allows you to train with just a `.yaml` file.

## Step 2.1. Setup Config:

The input into our training script is a `.yaml` file that contains all the parameters needed for training. Please checkout the [`README`](biogtr/training/configs/README.md) in `biogtr/training/configs` for a description of all the different parameters. We also provide an example config in [`biogtr/training/configs/base.yaml`](biogtr/training/configs/base.yaml) to give you an idea for how the config should look. In general, the best practice is to keep a single `base.yaml` file which has all the default arguments you'd like to use. Then you can have a second `.yaml` file which will override only those specific set of parameters when training (for an example see [`biogtr/training/configs/params.yaml`](biogtr/training/configs/params.yaml)).

## Step 2.2 Train Model

Once you have your config file and dataset set up, training is as easy as running

```bash
python /path/to/biogtr/training/train.py --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```
where `CONFIG_DIR` is the directory that `hydra` should search for the `config.yaml` and `CONFIG_STEM` is the name of the config without the `.yaml` extension.

e.g. If I have a config file called `base.yaml` inside my `/home/aaprasad/biogtr_configs` directory I can call
```bash
python /home/aaprasad/biogtr/training/train.py --config-base=/home/aaprasad/biogtr_configs --config-name=base
```

### Overriding Arguments
Instead of changing the `base.yaml` file every time you want to run a different config, `hydra` enables us to either 
1. provide another `.yaml` file with a subset of the parameters to overide
2. provide the args to the cli directly
#### File-based override
For overriding specific params with a sub-config, you can specify a `params_config` key and path in your `config.yaml` or run

```bash
python /path/to/biogtr/training/train.py --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] ++params_config="/path/to/params.yaml"
```

e.g. If I have a `params_to_override.yaml` file inside my `/home/aaprasad/biogtr_configs` directory that contains a only a small selection of parameters that I'd like to override, I can run:

```bash
python /home/aaprasad/biogtr/training/train.py --config-base=/home/aaprasad/biogtr_configs --config-name=base ++params_config=/home/aaprasad/biogtr_configs/params_to_override.yaml
```

#### CLI-based override
For directly overriding a specific param via the command line directly you can use the `section.param=key` syntax as so:

```bash
python /path/to/biogtr/training/train.py --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] section.param=value
```

"""
Note: Generally you will only need the direct override or the file-based override however you can technically do both via 
```bash
python /path/to/biogtr/training/train.py --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] ++params_config="/path/to/params.yaml" section.param=value
```
but you do need the `--config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM]` arguments regardless.
"""

e.g If now I want to override a couple parameters again, say change the number of attention heads and change the name of this run in my logger, I can pass `model.head=3` and `logger.name="test_nheads=3"` into 

```bash
python /home/aaprasad/biogtr/training/train.py --config-base=/home/aaprasad/biogtr_configs --config-name=base model.nhead=3 logger.name="test_nheads=3"
```

or alongside the `params_to_override.yaml`

```bash
python /home/aaprasad/biogtr/training/train.py --config-base=/home/aaprasad/biogtr_configs --config-name=base ++params_config=/home/aaprasad/biogtr_configs/params_to_override.yaml model.nhead=3 logger.name="test_nheads=3"
```
"""
Note: When overriding the parameters, make sure your config contains that parameter and you match the parameter names exactly otherwise it will cause an error.
"""
See [here](https://hydra.cc/docs/advanced/override_grammar/basic/) for more information on overriding params.

## Step 3. Inference

The output of the train script will be a `*.ckpt` file. You can load a model programmatically by running:

```python
from biogtr.models import GTRRunner

ckpt = "/path/to/model/*.ckpt"
runner = GTRRunner.load_from_ckpt(ckpt)
model = runner.model
```

which will give you a pretrained `GlobalTrackingTransformer` model.

### Step 3.1 Set up config

Similar to training, we need to set up a config file that specifies 
1. a `ckpt_path`
2. a `out_dir`
3. a `Tracker` config
4. a `dataset` config with a `test_dataset` subsection containing dataset params.

Please see the [README](biogtr/inference/configs/inference.yaml) in `biogtr/inference/configs` for details and see [`biogtr/inference/configs/inference.yaml`](biogtr/inference/configs/inference.yaml) for an example inference config file.

### Step 3.2 Run Inference

Just like training we can use the hydra syntax for specifying arguments via the cli. Because there aren't as many parameters here we recommend just overriding specific params via the cli directly but you're still more than welcome to use a subconfig yaml to override. Thus you can run inference via:

```bash
python /path/to/biogtr/inference/inference.py --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```

e.g. If I had an inference config called `track.yaml` inside `/home/aaprasad/biogtr_configs` 

```bash
python /home/aaprasad/biogtr/inference/inference.py --config-base=/home/aaprasad/biogtr_configs --config-name=track
```

"""
Note: you can use relative paths as well but may be a bit riskier so we recommend absolute paths whenever possible.
"""

This will run inference on the videos/detections you specified in the `dataset.test_dataset` section of the config and save the tracks to individual `/outdir/[VID_NAME].biogtr_inference.slp` files. Now you can load the file with `sleap-io` and do what you please!