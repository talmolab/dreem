### From raw tiff stacks to tracked identities

This notebook will walk you through the typical workflow for microscopy identity tracking. We start with an off-the-shelf detection model, and feed those results into DREEM. 
Here, we'll use the API, but we also provide a CLI interface for convenience.

To run this demo, we have provided sample data, model checkpoints, and configurations. The data used in this demo is small enough to be run on a single machine, though a GPU is recommended. 

#### Directory structure (data, models and configs will be downloaded)
```bash
./data
    /dynamicnuclearnet
        /test_1
        /mp4-for-visualization
    /lysosomes
        /7-2
        /7-2_GT
        /mp4-for-visualization
./configs
    sample-eval-microscopy.yaml
./models
    pretrained_microscopy.ckpt
 microscopy-demo-full-api.ipynb
```

#### Install huggingface hub to access models and data


```python
!pip install huggingface_hub
```

#### Import necessary packages


```python
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from omegaconf import OmegaConf
from dreem.io import Config
from dreem.datasets import TrackingDataset
from dreem.models import GTRRunner
from dreem.inference import Tracker
import sleap_io as sio
import matplotlib.pyplot as plt
import h5py
from huggingface_hub import hf_hub_download
```

#### Download a pretrained model, configs and some data


```python
model_save_dir = "./models"
config_save_dir = "./configs"
data_save_dir = "./data"
os.makedirs(config_save_dir, exist_ok=True)
os.makedirs(data_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
```


```python
model_path = hf_hub_download(repo_id="talmolab/microscopy-pretrained", filename="pretrained-microscopy.ckpt",
local_dir=model_save_dir)

config_path = hf_hub_download(repo_id="talmolab/microscopy-pretrained", filename="sample-eval-microscopy.yaml",
local_dir=config_save_dir)
```


```python
!huggingface-cli download talmolab/microscopy-demo --repo-type dataset --local-dir ./data
```

#### Verify that the model loads properly


```python
m = GTRRunner.load_from_checkpoint(model_path, strict=False)
```

Check if a GPU is available. For Apple silicon users, you can run on MPS, but ensure your version of PyTorch is compatible with MPS, and that you have installed the correct version of DREEM. You can also run without a GPU. The demo has been tested on an M3 Macbook Air running only on a CPU.



```python
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    accelerator = "cuda"
elif torch.backends.mps.is_available():
    accelerator = "mps"
    devices = 1
else:
    accelerator = "cpu"
print("Using device: ", accelerator)

torch.set_float32_matmul_precision("medium")
```

## Detection

Here we use CellPose to create segmentation masks for our instances. If you want to skip this stage, we have provided segmentation masks for the lysosomes dataset located at ./data/lysosomes. You can enter this path in the configuration file provided, under dataset.test_dataset.dir.path, and then skip straight ahead to the section labelled DREEM Inference below

#### Install CellPose


```python
!pip install git+https://www.github.com/mouseland/cellpose.git
```


```python
import tifffile
from cellpose import models
```


```python
data_path = "./data/dynamicnuclearnet/test_1"
segmented_path = "./data/dynamicnuclearnet/test_1_GT/TRA"
os.makedirs(segmented_path, exist_ok=True)
```

Set the approximate diameter (in pixels) of the instances you want to segment


```python
diam_px = 25
```

#### Run detection model


```python
tiff_files = [f for f in os.listdir(data_path) if f.endswith('.tif') or f.endswith('.tiff')]
stack = np.stack([tifffile.imread(os.path.join(data_path, f)) for f in tiff_files])
frames, Y, X = stack.shape

channels = [0, 0]
# use builtin latest model
model = models.CellposeModel(gpu=True)
all_masks = np.zeros_like(stack)
for i, img in enumerate(stack):
    masks, flows, styles = model.eval(
        img,
        diameter=diam_px,
        cellprob_threshold=0.0,
        channels=channels,
        z_axis=None,
    )
    all_masks[i] = masks
```

#### Save the segmentation masks


```python
os.makedirs(segmented_path, exist_ok=True)
for i, (mask, filename) in enumerate(zip(all_masks, tiff_files)):
    new_tiff_path = os.path.join(segmented_path, f"{os.path.splitext(filename)[0]}.tif")
    print(f"exporting frame {i} to tiff at {new_tiff_path}")
    tifffile.imwrite(new_tiff_path, mask)
```

### View the segmentation result and original image 


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(all_masks[0])
ax1.set_title('Segmentation Mask')
ax2.imshow(stack[0])
ax2.set_title('Original Image')
plt.tight_layout()
plt.show()
```

## DREEM Inference
In this section, we demonstrate the standard DREEM inference pipeline using the API


```python
model = GTRRunner.load_from_checkpoint(model_path, strict=False)
```

### Setup inference configs
#### NOTE: We can only specify 1 directory at a time when running inference. To test a different dataset, just enter the path to the directory containing the dataset. See the config for an example


```python
pred_cfg_path = "./configs/sample-eval-microscopy.yaml"
# use OmegaConf to load the config
pred_cfg = OmegaConf.load(pred_cfg_path)
pred_cfg = Config(pred_cfg)
```

Get the tracker settings from the config and initialize the tracker


```python
tracker_cfg = pred_cfg.get_tracker_cfg()
model.tracker_cfg = tracker_cfg
model.tracker = Tracker(**model.tracker_cfg)
trainer = pred_cfg.get_trainer()
# inference results will be saved here
outdir = "./results"
os.makedirs(outdir, exist_ok=True)
```

### Prepare data and run inference


```python
labels_files, vid_files = pred_cfg.get_data_paths(mode="test", data_cfg=pred_cfg.cfg.dataset.test_dataset)

for label_file, vid_file in zip(labels_files, vid_files):
    dataset = pred_cfg.get_dataset(
        label_files=[label_file], vid_files=[vid_file], mode="test"
    )
    dataloader = pred_cfg.get_dataloader(dataset, mode="test")
    
    # the actual inference is done here
    preds = trainer.predict(model, dataloader)

    # convert the predictions to sleap format
    pred_slp = []
    tracks = {}
    for batch in preds:
        for frame in batch:
            if frame.frame_id.item() == 0:
                video = (
                    sio.Video(frame.video)
                    if isinstance(frame.video, str)
                    else sio.Video
                )
            lf, tracks = frame.to_slp(tracks, video=video)
            pred_slp.append(lf)
    pred_slp = sio.Labels(pred_slp)
    # save the predictions to disk (requires sleap-io)
    if isinstance(vid_file, list):
        save_file_name = vid_file[0].split("/")[-2]
    else:
        save_file_name = vid_file
    outpath = os.path.join(
        outdir,
        f"{Path(save_file_name).stem}.dreem_inference.{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.slp",
    )
    pred_slp.save(outpath)
```

## Visualize the results


```python
import cv2
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.cm as cm
from IPython.display import HTML, display
import io
import base64
from IPython.display import Video

def create_tracking_animation(video_path, metadata_df, 
                                             fps=30, text_size=8, marker_size=20,
                                             max_frames=None, display_width=800):
    """
    Create and display an animal tracking animation directly in the notebook.
    
    Parameters:
    -----------
    video_path : str
        Path to the input MP4 video file
    metadata_df : pandas.DataFrame
        DataFrame with columns: frame_id, track_id, centroid
    fps : int
        Frames per second for the animation
    text_size : int
        Size of the ID text
    marker_size : int
        Size of the marker circle
    max_frames : int, optional
        Maximum number of frames to process (useful for previewing)
    display_width : int
        Width of the displayed animation in the notebook
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a colormap for track IDs
    unique_ids = metadata_df['track_id'].unique()
    cmap = cm.get_cmap('viridis', len(unique_ids))  # Using 'hsv' for bright, distinct colors
    id_to_color = {id_val: cmap(i) for i, id_val in enumerate(unique_ids)}
    
    # Set up the figure and axis with the correct aspect ratio
    fig_width = display_width / 100  # Convert to inches (assuming 100 dpi)
    fig_height = fig_width * (height / width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Initialize the plot elements
    frame_img = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    markers = []
    texts = []
    
    # Get the list of frame IDs from the metadata
    frame_ids = sorted(metadata_df['frame_id'].unique())
    
    # Limit the number of frames if specified
    if max_frames is not None and max_frames < len(frame_ids):
        frame_ids = frame_ids[:max_frames]
        print(f"Limiting preview to {max_frames} frames")
    
    # Function to update the animation for each frame
    def update(frame_num):
        # Read the frame from the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_num}")
            return []
        
        # Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img.set_array(frame_rgb)
        
        # Clear previous markers and texts
        for marker in markers:
            marker.remove()
        markers.clear()
        
        for text in texts:
            text.remove()
        texts.clear()
        
        # Get data for the current frame
        frame_data = metadata_df[metadata_df['frame_id'] == frame_num]
        
        # Add markers and IDs for each animal in the current frame
        for _, row in frame_data.iterrows():
            track_id = row['track_id']
            x, y = row['centroid']
            color = id_to_color[track_id]
            
            # Add circle marker
            circle = Circle((x, y), marker_size, color=color, alpha=0.3)
            markers.append(ax.add_patch(circle))
            
            # Add ID text
            text = ax.text(x, y, str(track_id), color='white', 
                          fontsize=text_size, ha='center', va='center', 
                          fontweight='bold')
            texts.append(text)
        
        # Add frame number for reference
        frame_text = ax.text(10, 20, f"Frame: {frame_num}", color='white', 
                            fontsize=text_size, backgroundcolor='black')
        texts.append(frame_text)
        
        return [frame_img] + markers + texts
    
    # Set up the axis
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
    ax.axis('off')
    
    # Create the animation
    print(f"Creating animation with {len(frame_ids)} frames...")
    anim = FuncAnimation(fig, update, frames=frame_ids, blit=True)
    
    # Display the animation in the notebook
    plt.close(fig)  # Prevent duplicate display
    
    # Display as HTML5 video
    html_video = HTML(anim.to_html5_video())
    display(html_video)
    
    return anim

# Option to save the animation to a file for later viewing
def save_animation(anim, output_path, fps=10, dpi=100):
    """Save the animation to a file"""
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)
    print(f"Animation saved to {output_path}")
    
    # Display the saved video in the notebook
    return Video(output_path, embed=True, width=800)
```

Load the predictions into a dataframe to make an animation


```python
list_frames = []
for lf in pred_slp:
    for instance in lf.instances:
        centroid = np.nanmean(instance.numpy(), axis=0)
        track_id = int(instance.track.name)
        list_frames.append({'frame_id': lf.frame_idx, 'track_id': track_id, 'centroid': centroid})
df = pd.DataFrame(list_frames)
```

Create and display the animation in the notebook


```python
for file in os.listdir(os.path.join(pred_cfg.cfg.dataset.test_dataset['dir']['path'], 'mp4-for-visualization')):
    if file.endswith('.mp4'):
        video_path = os.path.join(pred_cfg.cfg.dataset.test_dataset['dir']['path'], 'mp4-for-visualization', file)

anim = create_tracking_animation(
    video_path=video_path,
    metadata_df=df,
    fps=15,
    text_size=5,
    marker_size=8,
    max_frames=300
)

# save the animation
video = save_animation(anim, f"./tracking_vis-{video_path.split('/')[-1]}")
```

## Evaluate the tracking results
In this section, we evaluate metrics on a ground truth labelled test set. Note that in this example, the test set we used to demonstrate the inference pipeline is the same as the one we use here. To verify that
we do not in fact use any ground truth information during tracking, go to our full dreem-demo notebook, where we use de-labelled data to verify this


```python
pred_cfg_path = "./configs/eval.yaml"
# use OmegaConf to load the config
eval_cfg = OmegaConf.load(pred_cfg_path)
eval_cfg = Config(eval_cfg)
```


```python
model.metrics["test"] = eval_cfg.get("metrics", {}).get("test", "all")
model.test_results["save_path"] = eval_cfg.get("outdir", "./eval")
os.makedirs(model.test_results["save_path"], exist_ok=True)
```

Run evaluation pipeline. Note how we use trainer.test() to run evaluation whereas earlier, we used trainer.predict() to run inference


```python
labels_files, vid_files = eval_cfg.get_data_paths(mode="test", data_cfg=eval_cfg.cfg.dataset.test_dataset)
trainer = eval_cfg.get_trainer()
for label_file, vid_file in zip(labels_files, vid_files):
    dataset = eval_cfg.get_dataset(
        label_files=[label_file], vid_files=[vid_file], mode="test"
    )
    dataloader = eval_cfg.get_dataloader(dataset, mode="test")
    metrics = trainer.test(model, dataloader)
```

### Extract the results and view key metrics
The results get saved to an HDF5 file in the directory specified in the config


```python
for file in os.listdir(model.test_results["save_path"]):
    if file.endswith(".h5"):
        h5_path = os.path.join(model.test_results["save_path"], file)
```


```python
dict_vid_motmetrics = {}
dict_vid_gta = {}
dict_vid_switch_frame_crops = {}

with h5py.File(h5_path, "r") as results_file:
    # Iterate through all video groups
    for vid_name in results_file.keys():
        print("Extracting metrics and crops for video: ", vid_name)
        vid_group = results_file[vid_name]
        # Load MOT summary
        if "mot_summary" in vid_group:
            mot_summary_keys = list(vid_group["mot_summary"].attrs)
            mot_summary_values = [vid_group["mot_summary"].attrs[key] for key in mot_summary_keys]
            df_motmetrics = pd.DataFrame(list(zip(mot_summary_keys, mot_summary_values)), columns=["metric", "value"])
            dict_vid_motmetrics[vid_name] = df_motmetrics
        # Load global tracking accuracy if available
        if "global_tracking_accuracy" in vid_group:
            gta_keys = list(vid_group["global_tracking_accuracy"].attrs)
            gta_values = [vid_group["global_tracking_accuracy"].attrs[key] for key in gta_keys]
            df_gta = pd.DataFrame(list(zip(gta_keys, gta_values)), columns=["metric", "value"])
            dict_vid_gta[vid_name] = df_gta
        # Find all frames with switches and save the crops
        frame_crop_dict = {}
        for key in vid_group.keys():
            if key.startswith("frame_"):
                frame = vid_group[key]
                frame_id = frame.attrs["frame_id"]
                for key in frame.keys():
                    if key.startswith("instance_"):
                        instance = frame[key]
                        if "crop" in instance.keys():
                            frame_crop_dict[frame_id] = instance["crop"][:].squeeze().transpose(1,2,0)
        dict_vid_switch_frame_crops[vid_name] = frame_crop_dict
```

Check the switch count (and other mot metrics) for the whole video


```python
motmetrics = list(dict_vid_motmetrics.values())[0]
# motmetrics.loc[motmetrics['metric'] == 'num_switches']
motmetrics
```

Check global tracking accuracy. This represents the percentage of frames where the tracker correctly maintained identities for each instance.


```python
gta = list(dict_vid_gta.values())[0]
gta
```
