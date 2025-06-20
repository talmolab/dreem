# Quickstart

DREEM operations can be performed either through the command-line interface or through the API (for more advanced users). The CLI provides all the commands for the main functionalities from training a model to running eval and inference to generating visualizations. All operations can be customized via configs that are managed by [`hydra`](https://hydra.cc). A more in depth explanation of using DREEM, See the [User Guide](./usage.md). For a complete reference of all commands and options, see [the API Reference](https://dreem.sleap.ai/reference/dreem/). To quickly test your installation and familiarize yourself with `dreem`, you can follow the fly tracking example.

## Fly tracking example.

In this example we will track a social interaction between two flies from [sleap's fly32 dataset](https://sleap.ai/datasets.html#fly32) using a pretrained model. This example assumes that you have a conda environment installed with the dreem package. Please see [the installation guide](./index.md#installation) if you haven't installed it yet.

### Get data
First, we need at least one video and a set of corresponding detections to work with. For this example, we provide a video and a `.slp` file with pose keypoints for the video. This dataset can be downloaded using [`gdown`](https://github.com/wkentaro/gdown). First, let's first move into our home directory via 

```bash
cd ~
```

Next, make sure you have `gdown` installed (it should be installed in the conda env already if you followed the installation guide):

```bash
pip install gdown
```

Now download the dataset using `gdown`

```bash
gdown --fuzzy https://drive.google.com/file/d/1grmoUH8ugDIF3z9djbDfu0sx8ylwPErB/view
```

This should download a `.zip` file called `demo-assets.zip` to your current working directory which you can confirm by running

```bash
ls demo-assets.zip
```

which should output `demo-assets.zip`. Now, we just need to unzip the folder with 

```bash
unzip demo-assets.zip
```

To confirm everything is correct, you can run

```bash
ls -R demo-assets
```

which will output:
```
base.yaml               inference.yaml          val
dreem-demo.ipynb        test
eval.yaml               train

demo-assets/test:
190719_090330_wt_18159206_rig1.2@15000-17560.mp4
GT_190719_090330_wt_18159206_rig1.2@15000-17560.slp

demo-assets/train:
190612_110405_wt_18159111_rig2.2@4427.mp4
GT_190612_110405_wt_18159111_rig2.2@4427.slp

demo-assets/val:
GT_two_flies.slp        two_flies.mp4
```

We'll also use this as our working directory so let's move into it.
```bash
cd demo-assets
```

To summarize this into one set of commands, we simply need to run:
```bash
cd ~ # move to home directory
pip install gdown # make sure `gdown` is installed
gdown --fuzzy https://drive.google.com/file/d/1grmoUH8ugDIF3z9djbDfu0sx8ylwPErB/view #download zip file
unzip demo-assets.zip # unzip compressed version
cd demo-assets # move into demo assets as working directory
```

### Get model checkpoint.
Now we'll pull a pretrained model checkpoint trained on flies in order to run inference from [`huggingface`](https://huggingface.co/talmolab/animals-pretrained). First make sure you have [the `huggingface` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) installed (once again it should be installed in your conda environment already)

```bash
pip install "huggingface_hub[cli]"
```

Now we download the pretrained animal tracking model via
```bash
huggingface-cli download talmolab/animals-pretrained animals-pretrained.ckpt --local-dir=.
```

To confirm this downloaded properly you can once again run
```bash
ls animals-pretrained.ckpt
```
which should output

```bash
animals-pretrained.ckpt
```

To summarize, run:
```bash
pip install "huggingface_hub[cli]" # make sure hugging-face cli is installed
huggingface-cli download talmolab/animals-pretrained animals-pretrained.ckpt --local-dir=. #download checkpoint
ls animals-pretrained.ckpt #confirm checkpoint is downloaded
```

### Run Tracking

Finally, we can run tracking quite easily via

```bash
dreem-track --config-dir=. --config-name=eval ckpt_path=./animals-pretrained.ckpt  #TODO: make sure ckpt file is correct 
```

Once completed, it should output a file in the `eval` folder called `GT_190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.slp`
which you can confirm by running

```bash
ls eval/GT_190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.slp
```

## What's next?
### Visualize Results
First, we recommend visualizing the outputs of the tracks you just made. You can do so by first installing sleap via [its installation guide](https://sleap.ai/#quick-install) and then running

```bash
sleap-label eval/GT_190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.slp
```

### Run through full workflow 
Next, We recommend checking out [the full usage guide](./usage.md) and try to run through the whole work flow of

1. Training a model
2. Evaluating the model on held out data
3. Running inference on another held out video
4. Visualizing the results.

This will get you very comfortable with configuring custom training jobs, evaluating your models, tracking your own data and visualizing the results.

### (Optional) Try out API Demo
If you're comfortable with python (more specifically pytorch), You can also check out our demo notebook for how to use the API to work through the above workflow instead. This will enable you to do more custom things with DREEM!.


### Get some SLEAP and sweet DREEMs!

The animal checkpoint we used above was actually pretrained on a wide variety of animals from mice, to flies, to zebrafish. Thus, you can actually just generate some detection data via SLEAP and then use our pretrained model as we just did to run tracking!

