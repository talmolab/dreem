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

This repo uses [`hydra`](https://hydra.cc) for config handling, and [`pytorch`](https://pytorch.org)/[`pytorch-lightning`](https://lightning.ai) to handle high-level training/eval/inference. Thus, we recommend skimming thru their respective docs to get some familiarity but is not necessary.

## Step 1. Data Acquisition.

### Step 1.1: Get detections
One of the main advantages of this system is that we ***decouple** detection and tracking. This means that our model assumes that you already have detections available when you are ready to track. In order to get these detections we recommend a couple methods. For animal pose-estimation, we highly recommend heading over to [`sleap`](https://sleap.ai) and running through their work flow. For microscopy tracking, checkout `trackmate`(https://imagej.net/plugins/trackmate/) as well as `cellpose`(https://www.cellpose.org) and `StarDist`(https://imagej.net/plugins/stardist). If you are only doing tracking (ie inference), then you once you have your detections you are good to go! 
### ***FOR TRAINING*** Step 1.2 Proofreading
Otherwise, we recommend using the `sleap-label` gui to proofread your track labels. For animal tracking, if you used `sleap` you can start proofreading right away. Otherwise if you used a different system (e.g `DeepLabCut`) check out `sleap.io.format` for available converters. With microscopy, we highly recommend starting out with `trackmate` and then proofreading in `sleap`. Here is [a converter from trackmate's output to a `.slp`] file. In general, you can use `sleap-io`(https://io.sleap.ai/latest/) to write a custom converter to `.slp` if you'd like to use the sleap-gui for proofreading
### Step 1.3 Organize data.

Although, our data loading does take paths to label files and video files directly so it's fairly flexible, we recommend organizing your data with a couple things in mind.

1. Match video and labels file stems. Because our datasets just take in a list of video files and a list of label files, the order in which the corresponding files are passed must match. (e.g `[file1.slp, file2.slp], [vid1.mp4, vid2.mp4]`. In order to make programmitc file searching easy its best to save your labels and vid files with the same stem so that you can ensure the ordering will be consistent. Its also just best practice so you know which video a labels file corresponds to.
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

The input into our training script is a `.yaml` file that contains all the parameters needed for training. Please checkout the `README` in `biogtr/training/configs` for a description of all the different parameters. We also provide an example config in `biogtr/training/configs/base.yaml` to give you an idea for how the config should look. In general, the best practice is to keep a single `base.yaml` file which has all the default arguments you'd like to use. Then you can have a second `.yaml` file which will override only those specific set of parameters when training.

## Step 2.2 Train Model

Once you have your config file and dataset set up, training is as easy as running

```bash
python train.py --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```