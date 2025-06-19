# Quickstart

DREEM operations can be performed either through the command-line interface or through the API (for more advanced users). The CLI provides all the commands for the main functionalities from training a model to running eval and inference to generating visualizations. All operations can be customized via configs that are managed by [`hydra`](https://hydra.cc). A more in depth explanation of using DREEM, See the [User Guide](./usage.md). For a complete reference of all commands and options, see [the API Reference](https://dreem.sleap.ai/reference/dreem/). To quickly test your installation and familiarize yourself with `dreem`, you can follow the fly tracking example.

## Fly tracking example.

In this example we will track a social interaction between two flies from [sleap's fly32 dataset](https://sleap.ai/datasets.html#fly32) using a pretrained model. This example assumes that you have a conda environment installed with the dreem package. Please see [the installation guide](./index.md#installation) if you haven't installed it yet.

### Get data
First, we need at least one video and a set of corresponding detections to work with. For this example, we provide a video and a `.slp` file with pose keypoints for the video. This dataset can be downloaded using [`gdown`](https://github.com/wkentaro/gdown). First make sure you have `gdown` installed (it should be installed in the conda env already if you followed the installation guide):

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

To summarize this into one set of commands, we simply need to run:
```bash
pip install gdown # make sure `gdown` is installed
gdown --fuzzy https://drive.google.com/file/d/1grmoUH8ugDIF3z9djbDfu0sx8ylwPErB/view #download zip file
unzip demo-assets.zip # unzip compressed version
```

### Get model checkpoint.
Now we'll pull a pretrained model checkpoint trained on flies in order to run inference from [huggingface](https://huggingface.co/talmolab/animals-pretrained) via the following command

```bash
git lfs install ## Make sure git-lfs is installed (https://git-lfs.com)
git clone https://huggingface.co/talmolab/animals-pretrained
```

To confirm this downloaded properly you can once again run
```bash
ls animals-pretrained
```
which should output

```bash
animals-base.yaml               sample-eval-animals.yaml
animals-override.yaml
```

### Run Tracking

Finally, we can run tracking quite easily via

```
dreem-track --config-dir=. --config-name=eval ckpt_path=./animals-pretrained/pretrained-animals.ckpt  #TODO: make sure ckpt file is correct 
```

Once completed, it should output a file called `GT_190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.slp`
which you can confirm by running

```bash
ls GT_190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.slp
```

### Visualize Tracks

Finally, we can visualize the output of dreem via `dreem-visualize` by running the following command:

```bash
dreem-visualize +labels_path=./GT_190719_090330_wt_18159206_rig1.2@15000-17560.dreem_inference.slp +save_path=./demo_annotation.mp4
```

This will output an mp4 file of the animal behavior video annotated by the bounding box which have been colored by track id.

## What's next?
First, we recommend checking out the full usage guide and try to run through the whole work flow of 
1. Training a model
2. Evaluating the model on held out data
3. Running inference on another held out video
4. Visualizing the results.


If you're comfortable with python (more specifically pytorch), You can also check out our demo notebook for how to use the API to work through the above workflow instead. This will enable you to do more custom things with DREEM!.

Now, get some SLEAP and sweet DREEMs!

