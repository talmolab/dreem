# Usage

This page gives detailed instructions for using DREEM. We also have a [quickstart guide](./quickstart.md) and notebooks in the Examples section to help you get started.

<!-- ## Background

This repo uses 3 main packages:

1. [`hydra`](https://hydra.cc) for configurations
2. [`pytorch`](https://pytorch.org) for model construction
3. [`pytorch-lightning`](https://lightning.ai) for high-level training/eval/inference -->

## Installation

Head over to the [installation guide](./installation.md) to get started.

## Training

DREEM enables you to train your own model based on your own annotated data. This can be useful when the pretrained models, or traditional approaches to tracking don't work well for your data.

### Generate Ground Truth Data
To train a model, you need:

1. A video
    - For animal data see the [`imageio`](https://imageio.readthedocs.io/en/v2.4.1/formats.html) docs for supported file types. Common ones include mp4, avi, etc.
    - For microscopy data we currently support `.tif` stacks.
2. A ground truth labels file in [SLEAP](https://sleap.ai) or [CellTrackingChallenge](https://celltrackingchallenge.net) format. This labels file must contain:
    1. Detections (i.e. locations of the instances in each frame). This can come in the form of centroids or pose keypoints for SLEAP format data, or segmentation masks for Cell Tracking Challenge format data.
    2. Ground truth identities (also called tracks). These are temporally consistent labels that link detections across time.

#### Get Initial labels
To generate your initial labels we recommend a couple methods:

 - For animal tracking, we recommend using [SLEAP](https://sleap.ai). SLEAP provides a graphical user interface that makes it easy to annotate data from scratch, and output the labels file in the SLEAP format.
 - For microscopy tracking, check out [CellPose](https://www.cellpose.org) or [Ilastik](https://www.ilastik.org). These methods output segmentation masks, but do not provide tracks. [Fiji](https://imagej.net/software/fiji/) offers several end-to-end segmentation and tracking options. Recall that your labels file must contain tracks.

#### Proofreading
Once you have your labels file containing initial detections and tracks, you'll want to [proofread](https://sleap.ai/tutorials/proofreading.html#track-proofreading) your labels. Obtaining good results relies on having accurate ground truth tracks. The annotated data should follow these guidelines:

1. No identity switches. This is important for training a model that maintains temporally consistent identities.

2. Good detections. Since the input to the model is a crop centered around each detection, we want to make sure the coordinates we crop around are as accurate as possible.

We recommend using the [`sleap-label` GUI](https://sleap.ai/guides/gui.html) for [proofreading](https://sleap.ai/guides/proofreading.html#id1). SLEAP provides tools that make it easy to correct errors in tracking.

##### Converting data to a SLEAP compatible format
In order to use the SLEAP GUI you'll need to have your labels and videos in a SLEAP compatible format. Check out [the sleap-io docs](https://io.sleap.ai/latest/formats/) for available formats. The easiest way to ensure your labels are compatible with sleap is to convert them to a `.slp` file. Otherwise if you used a different system (e.g DeepLabCut) check out [`sleap.io.convert`](https://sleap.ai/api/sleap.io.convert.html#module-sleap.io.convert) for available converters. With microscopy, we highly recommend starting out with TrackMate and then proofread in SLEAP's gui. Here is [a converter from trackmate's output to a `.slp`](https://gist.github.com/aaprasad/5243be0785a40e9dafa1697ce2258e3e) file. In general, you can use [`sleap-io`](https://io.sleap.ai/latest/) to write a custom converter to `.slp` if you'd like to use the sleap-gui for proofreading.

#### Organize data

For animal tracking, you'll need video/slp pairs in a directory that you can specify in the configuration files when training. For instance, you can have separate train/val/test directories, each with slp/video pairs. The naming convention is not important, as the .slp labels file has a reference to its associated video file when its created.

For microscopy tracking, you'll need to organize your data in the Cell Tracking Challenge format. We've provided a sample directory structure below. Check out this [guide](https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf) for more details.

Using the **SLEAP** format:
```
dataset_name/
    train/
        vid_1.{VID_EXTENSION}
        vid_1.slp
            .
            .
            .
        vid_n.{VID_EXTENSION}
        vid_n.slp
    val/
        vid_1.{VID_EXTENSION}
        vid_1.slp
            .
            .
            .
        vid_n.{VID_EXTENSION}
        vid_n.slp
    test/ # optional; test sets are not automatically evaluated as part of training
        vid_1.{VID_EXTENSION}
        vid_1.slp
            .
            .
            .
        vid_n.slp
        vid_n.slp
```
The **CellTrackingChallenge** format requires a directory with raw tifs, and a matching directory with labelled segmentation masks for the track labels. The directory structure is as follows:
```
dataset_name/
    train/
        subdir_0/
            frame0.tif # these are raw images
            ...
            frameN.tif
        subdir_0_GT/TRA # these are labelled segmentation masks
            frame0.tif
            ...
            frameN.tif
        subdir_1/
            frame0.tif
            ...
            frameN.tif
        subdir_1_GT/TRA
            frame0.tif
            ...
            frameN.tif
        ...
    val/
        subdir_0/
        subdir_0_GT/TRA
        subdir_1/
        subdir_1_GT/TRA
        ...
    test/ # optional; test sets are not automatically evaluated as part of training
        subdir_0/
        subdir_0_GT/TRA
        subdir_1/
        subdir_1_GT/TRA
        ...
```

### Training

Now that you have your dataset set up, let's start training a model! We provide a CLI that allows you to train with a simple command and a single yaml configuration file.

#### Setup Config

The input into our training script is a `.yaml` file that contains all the parameters needed for training. Please see [here](configs/training.md) for a detailed description of all the parameters and how to set up the config. We provide up-to-date sample configs on HuggingFace Hub in the [talmolab/microscopy-pretrained](https://huggingface.co/talmolab/microscopy-pretrained) and [talmolab/animals-pretrained](https://huggingface.co/talmolab/animals-pretrained). In general, the best practice is to keep a single `base.yaml` file which has all the default arguments you'd like to use. Then you can have a second `.yaml` file which will override a specific set of parameters when training.

#### Train Model

Once you have your config file and dataset set up, training is as easy as running

```bash
dreem-train --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```
where `CONFIG_DIR` is the directory that `hydra` should search for the `config.yaml` and `CONFIG_STEM` is the name of the config without the `.yaml` extension.

e.g. If you have a config file called `base.yaml` inside your `/home/user/dreem_configs` directory you can call
```bash
dreem-train --config-base=/home/user/dreem_configs --config-name=base
```

> Note: you can use relative paths as well but may be a bit riskier so we recommend absolute paths whenever possible.

If you've been through the example notebooks, you'll notice that training was done using the API rather than the CLI. You can use whichever you prefer.

##### Overriding Arguments
Instead of changing the `base.yaml` file every time you want to train a model using different configurations, `hydra` enables us to either

1. provide another `.yaml` file with a subset of the parameters to override
2. provide the args to the cli directly

We recommend using the file-based override for logging and reproducibility.

For overriding specific params with an override file, you can run:

```bash
dreem-train --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM] ++params_config="/path/to/override_params.yaml"
```

e.g. If you have a `override_params.yaml` file inside your `/home/user/dreem_configs` directory that contains a only a small selection of parameters that you'd like to override, you can run:

```bash
dreem-train --config-base=/home/user/dreem_configs --config-name=base ++params_config=/home/user/dreem_configs/override_params.yaml
```

<!-- ###### CLI-based override
For directly overriding a specific param via the command line directly you can use the `section.param=key` syntax as so:

```bash
dreem-train --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] section.param=value
```

e.g If now I want to override a couple parameters again, say change the number of attention heads and change the name of this run in my logger, I can pass `model.head=3` and `logger.name="test_nheads=3"` into 

```bash
dreem-train --config-base=/home/aaprasad/dreem_configs --config-name=base model.nhead=3 logger.name="test_nheads=3"
```
> Note: using the `section.param` syntax for CLI override will only override if the parameter exists in your config file otherwise an error will be thrown
> if you'd like add a new parameter you can add `++` to the front of `section.param` e.g `++model.nhead=3`, However in this case, if the parameter exists in the config it will throw an error.
> thus if you'd like to overwrite when the parameter exists, otherwise create a new one you can add a single `+` to the front of `section.param` e.g `+model.nhead=3`. 
>When using `+`, you'll want to make sure you've matched the param exactly, otherwise it will simply add a new parameter with your spelling and the original value won't change. 
> e.g doing `model.n_head=3` will cause the original `model.nhead` to stay the same and just add a `model.n_head` field.

See [here](https://hydra.cc/docs/advanced/override_grammar/basic/) for more information on overriding params.

> Note: You will usually only need the direct override or the file-based override however you can technically do both via 
> ```bash
> dreem-train --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] ++params_config="/path/to/params.yaml" section.param=value
> ```
> e.g. alongside the `params_to_override.yaml`
> ```bash
> dreem-train --config-base=/home/aaprasad/dreem_configs --config-name=base ++params_config=/home/aaprasad/dreem_configs/params_to_override.yaml model.nhead=3 logger.name="test_nheads=3"
> ```
> but the `--config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM]` arguments are always required. 
> However, be careful to ensure there are no conflicts when combining CLI and file-based override syntax
> (e.g make sure `section.param` doesn't appear both in your override file and in the CLI). 
> This is because hydra does not make a distinction, it will first overwrite with the CLI value and then overwrite with the params file value.
> this means that it will always default to the params file value when there are conflicts. -->

#### Output
The output of the train script will be at least 1 `ckpt` file, assuming you've configured the `checkpointing` section of the config correctly.

## Eval (with ground truth labels)
To test the performance of your model, you can use the `dreem-eval` CLI. It computes multi-object tracking metrics on your test data and outputs it in h5 format.

### Setup data
Note that your data should have ground truth labels. You can arrange it in a /test directory as shown above.

### Setup config

Samples are available at [talmolab/microscopy-pretrained](https://huggingface.co/talmolab/microscopy-pretrained) and [talmolab/animals-pretrained](https://huggingface.co/talmolab/animals-pretrained), while a detailed walkthrough is available [here](configs/inference.md)

### Run evaluation 

We provide a CLI that allows you to evaluate your model with a simple command and a single yaml configuration file.

```bash
dreem-eval --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```


<!-- #### Overriding Parameters.
Because there aren't as many parameters during inference as during training we recommend just using the cli-based override rather than the file based but you're more than welcome to do so.

In order to override params via the CLI, we can use the same `hydra` `section.param` syntax:

```bash
dreem-eval --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM] section.param=[VALUE]
```
e.g if I want to set the window size of the tracker to 32 instead of 8 through `tracker.window_size=32` and use a different model saved in `/home/aaprasad/models/new_best.ckpt` I can do:
```bash
dreem-eval --config-base=/home/aaprasad/dreem_configs --config-name=track ckpt_path="/home/aaprasad/models/new_best.ckpt" tracker.window_size=32`
``` -->
### Output
Tracking results will be saved as .slp, and the evaluation metrics will be saved in an hdf5 file saved to the `outdir` argument. We provide a script to extract the results from the h5 file, which you can find in the repository [here](https://github.com/talmolab/dreem/blob/main/scripts/extract_metrics.ipynb).

## Inference

In general, you will likely not have access to ground truth labels for your videos. In this case, you can use DREEM to run inference on your videos using a pretrained model or a model you trained yourself.

### Setup Data
#### Get detections
One of the main advantages of DREEM is that we *decouple* detection and tracking so you can use state of the art, off-the-shelf detection models. This means that in order to run inference (tracking) with our model you'll need the following:

1. A model saved as a `.ckpt` file
2. A video (same as in training)
3. A labels file. This labels file is slightly different than in training since we now only need detections but no ground truth tracks. So, you don't need to run a tracking method and proofread the results. You can simply run your detection model of choice and convert the data to SLEAP or CellTrackingChallenge format.

#### Organize Data

See the [training section](#organize-data) for more details on how to organize your data.

### Set up config

See the [eval section](#eval-with-ground-truth-labels) for more details on how to set up a config file. The format for eval and inference is the same.

### Run Inference

Similar to `dreem-eval`, you can run inference via:

```bash
dreem-track --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```
<!-- 

#### Overriding Parameters.
Because there aren't as many parameters during inference as during training we recommend just using the cli-based override rather than the file based but you're more than welcome to do so.

In order to override params via the CLI, we can use the same `hydra` `section.param` syntax:

```bash
dreem-track --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM] section.param=[VALUE]
```
e.g if I want to set the window size of the tracker to 32 instead of 8 through `tracker.window_size=32` and use a different model saved in `/home/aaprasad/models/new_best.ckpt` I can do:
```bash
dreem-track --config-base=/home/aaprasad/dreem_configs --config-name=track ckpt_path="/home/aaprasad/models/new_best.ckpt" tracker.window_size=32`
``` -->
### Output
Tracking results will be saved as .slp in the directory specified by the `outdir` argument. If you don't enter an `outdir` in the config, it will save to `./results`.

You should now be ready to use DREEM to train and track your own data!