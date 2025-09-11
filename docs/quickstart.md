# Quickstart

DREEM operations can be performed either through the command-line interface or through the API (for more advanced users). The CLI provides all the commands for the main functionalities from training a model to running eval and inference to generating visualizations. All operations can be customized via configs that are managed by Hydra. For a more in-depth walkthrough of DREEM, see the Examples section. For a complete reference of all commands and options, see [the API Reference](https://dreem.sleap.ai/reference/dreem/). 

To quickly test your installation and familiarize yourself with DREEM, you can follow the quickstart guide below.

## Fly tracking

In this example we will track a social interaction between two flies from the SLEAP [fly32 dataset](https://sleap.ai/datasets.html#fly32) using a pretrained model. This example assumes that you have a conda environment installed with the dreem package. Please see [the installation guide](./index.md#installation) if you haven't installed it yet.

### Download the data
First, we need at least one video and a set of corresponding detections to work with. For this example, we provide a video and a `.slp` file with pose keypoints for the video. This dataset can be downloaded from Hugging Face. The dataset includes configuration files needed for running inference.

First, make sure you have `huggingface-hub` installed (it should already be installed in your environment if you installed using the commands on the [installation page](./installation.md)):

```bash
pip install huggingface_hub
```

Download the dataset. The ```--local-dir``` flag allows you to specify the download location on your computer.

```bash
huggingface-cli download talmolab/sample-flies --repo-type dataset --local-dir ./data
```

### Download a model
Now we'll pull a pretrained model trained on various animal data in order to run inference.

```bash
huggingface-cli download talmolab/animals-pretrained animals-pretrained.ckpt --local-dir=./models
```

To confirm this downloaded properly you can run
```bash
ls animals-pretrained.ckpt
```

Your directory structure should look like this:
```bash
./data
    /test
        190719_090330_wt_18159206_rig1.2@15000-17560.mp4
        GT_190719_090330_wt_18159206_rig1.2@15000-17560.slp
    /train
        190612_110405_wt_18159111_rig2.2@4427.mp4
        GT_190612_110405_wt_18159111_rig2.2@4427.slp
    /val
        two_flies.mp4
        GT_two_flies.slp
    /inference
        190719_090330_wt_18159206_rig1.2@15000-17560.mp4
        190719_090330_wt_18159206_rig1.2@15000-17560.slp
    /configs
        inference.yaml
        base.yaml
        eval.yaml
./models
    animals-pretrained.ckpt
```

### Run Tracking

Tracking is easy to run using the CLI. Simply specify the path to the directory containing ```inference.yaml``` and the path to the model checkpoint.

```bash
dreem-track --config-dir=./data/configs --config-name=inference ckpt_path=./models/animals-pretrained.ckpt
```
If you want to evaluate the tracking accuracy, you can use the `dreem-eval` command, a drop-in replacement for `dreem-track` that outputs evaluation metrics and a detailed frame-by-frame multi-object tracking event log.
You can use it with this command. Note the different eval config file; this is only for illustrative purposes, to specify a test dataset path that has ground truth labels.

 ```bash
dreem-eval --config-dir=./data/configs --config-name=eval ckpt_path=./models/animals-pretrained.ckpt
```

Once completed, it should output a file in a new `results` folder called `GT_190719_090330_wt_18159206_rig1.2@15000-17560.<timestamp>dreem_inference.slp`

### Visualize Results
First, we recommend visualizing the outputs of the tracks you just made. You can do so by first installing sleap via [its installation guide](https://sleap.ai/#quick-install) and then running

```bash
sleap-label results/GT_190719_090330_wt_18159206_rig1.2@15000-17560.<timestamp>dreem_inference.slp
```
Note that the sleap-label command opens the SLEAP GUI which may not render on a remote server. 

### Check out the example notebooks
Once you're ready to dive deeper, head over to the Examples section and check out the notebooks there. Go through the end-to-end demo 
of the DREEM pipeline, from obtaining data to training a model, evaluating on a held-out dataset, and visualizing the results, all in the notebook. We also have
microscopy examples in which we use an off-the-shelf detection model to show how DREEM can be used with existing tools.

### Run through full workflow 
For more detail on the CLI, configuration files, and more, see the [Usage Guide](./usage.md)

### Get some SLEAP and sweet DREEMs!

The animal checkpoint we used above was trained on mice, flies and zebrafish. You can generate detections on raw videos via SLEAP and then use our pretrained model as we just did to run tracking.
