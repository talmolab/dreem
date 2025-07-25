# End-to-end demo

This notebook will walk you through the DREEM pipeline end to end, from obtaining data to training a model, evaluating on a held-out dataset, and visualizing the results. Here, we'll use the API, but we also provide a CLI interface for convenience.

To run this demo, we have provided sample data and configurations. The data used in this demo is small enough to be run on a single machine, though a GPU is recommended. 

#### Directory structure after downloading data:
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
```


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

## Download data and configs


```python
!huggingface-cli download talmolab/sample-flies --repo-type dataset --local-dir ./data
```

## Training

#### Setup configs
The configs provided are good defaults. You can change them as you see fit.


```python
config_path = "./data/configs/base.yaml"
# use OmegaConf to load the config
cfg = OmegaConf.load(config_path)
train_cfg = Config(cfg)
```

#### Create a model
The model is a Lightning wrapper around our model. Lightning simplifies training, validation, logging, and checkpointing.


```python
model = train_cfg.get_gtr_runner()
```

#### Prepare torch datasets and dataloader
Note: We use a batch size of 1 - we handle the batching ourselves since we are dealing with video data and associated labels in .slp format. The default clip length is set to 32 frames.


```python
train_dataset = train_cfg.get_dataset(mode="train")
train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

val_dataset = train_cfg.get_dataset(mode="val")
val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

# wrap the dataloaders
dataset = TrackingDataset(train_dl=train_dataloader, val_dl=val_dataloader)
```

#### Visualize the input data
The input data to the model is a set of crops taken around a particular keypoint on the instance. For animals, this can be any keypoint, and for microscopy, this is often the centroid. Augmentations are also applied. Since we shuffle the data, the frame ids you get may not be the first frames of the video.


```python
# load a batch of data in
viewer = iter(train_dataloader)
batch = next(viewer)
# save the crops for all frames in the batch
crops = {}
for frame in batch[0]:
    crops[frame.frame_id.item()] = []
    for instance in frame.instances:
        crops[frame.frame_id.item()].append(instance.crop.squeeze().permute(1,2,0).numpy())
```

Plot the crops for all frames in the batch


```python
total_crops = sum(len(crops) for crops in crops.items())

# Determine a grid size
n_cols = int(np.ceil(np.sqrt(total_crops)))
n_rows = int(np.ceil(total_crops / n_cols))

# Create figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(25,25))
fig.suptitle(f"Video: {frame.video}, Frames: {min(crops.keys())} to {max(crops.keys())}", fontsize=16)

# Ensure axes is always a 2D array
if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
elif n_rows == 1: axes = axes.reshape(1, -1)
elif n_cols == 1: axes = axes.reshape(-1, 1)

# Flatten for easier indexing
axes_flat = axes.flatten()

# Plot each crop
ax_idx = 0
for frame_id, vid_crops in sorted(crops.items()):
    for i, crop in enumerate(vid_crops):
        if ax_idx < len(axes_flat):
            ax = axes_flat[ax_idx]
            
            # Handle both RGB and grayscale images
            if crop.ndim == 3:
                # Normalize if needed
                if crop.max() > 1.0:
                    crop = crop / 255.0
                ax.imshow(crop)
            else:
                ax.imshow(crop, cmap='gray')
            
            ax.set_title(f"Frame {frame_id}, Inst {i}")
            ax.axis('off')
            ax_idx += 1

# Hide unused subplots
for i in range(ax_idx, len(axes_flat)):
    axes_flat[i].axis('off')

# Adjust layout to minimize whitespace
plt.tight_layout()
plt.subplots_adjust(top=0.95) 
```

#### Train the model
First setup various training features such as loss curve plotting, early stopping and more, through Lightning's callbacks, then setup the Trainer and train the model.


```python
# to plot loss curves 
class NotebookPlotCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        self.train_losses.append(train_loss.item())
        self.epochs.append(trainer.current_epoch)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        self.val_losses.append(val_loss.item())

notebook_plot_callback = NotebookPlotCallback()
```


```python
callbacks = []
_ = callbacks.extend(train_cfg.get_checkpointing())
_ = callbacks.append(pl.callbacks.LearningRateMonitor())
early_stopping = train_cfg.get_early_stopping()
if early_stopping is not None:
    callbacks.append(early_stopping)
callbacks.append(notebook_plot_callback)
```

The default maximum epochs is set to 4 in the provided config. You can change this in the trainer section of the config.


```python
# setup Lightning Trainer
trainer = train_cfg.get_trainer(
    callbacks,
    accelerator=accelerator,
    devices=1
)
# train the model
trainer.fit(model, dataset)
```

#### Visualize the train and validation loss curves


```python
plt.figure(figsize=(6,4))
plt.plot(notebook_plot_callback.epochs, notebook_plot_callback.train_losses, label='Train Loss', marker='o')
plt.plot(notebook_plot_callback.epochs, notebook_plot_callback.val_losses[1:], label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

## Inference
### Here we run inference on a video with **no** ground truth labels


```python
# get the model from the directory it saves to 
# (see logging.name in the config)
ckpt_dir = "./models/example_train"
ckpts = os.listdir(ckpt_dir)
for ckpt in ckpts:
    if "final" in ckpt: # assumes the final checkpoint is the best one
        best_checkpoint_path = os.path.join(ckpt_dir, ckpt)
        break
model = GTRRunner.load_from_checkpoint(best_checkpoint_path)
```

### Setup inference configs


```python
pred_cfg_path = "./data/configs/inference.yaml"
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
    outpath = os.path.join(
        outdir, f"{Path(label_file).stem}.dreem_inference.{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.slp"
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

def create_animal_tracking_animation_notebook(video_path, metadata_df, 
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
    cmap = cm.get_cmap('tab10', len(unique_ids))  # Using 'hsv' for bright, distinct colors
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
            circle = Circle((x, y), marker_size, color=color, alpha=0.7)
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
for file in os.listdir(pred_cfg.cfg.dataset.test_dataset['dir']['path']):
    if file.endswith('.mp4'):
        video_path = os.path.join(pred_cfg.cfg.dataset.test_dataset['dir']['path'], file)

anim = create_animal_tracking_animation_notebook(
    video_path=video_path,
    metadata_df=df,
    fps=15,
    text_size=8,
    marker_size=20,
    max_frames=300
)

# save the animation
video = save_animation(anim, f"./animal_tracking_vis-{video_path.split('/')[-1]}")
```

## Evaluate the tracking results
### Here we run inference on a video **with** ground truth labels. Then we will compute metrics for our tracking results. 
Note that we are only using separate configs for inference and evaluation so you can verify that the test file has no ground truth in it for inference, and that it does for evaluation. For eval, the slp file should have a "GT_" prefix to indicate that it is a ground truth file. For eval, you can also specify the metrics you want to compute. We offer a CLI interface for both evaluation and inference.


```python
pred_cfg_path = "./data/configs/eval.yaml"
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

Check the switch count (and other mot metrics) for the whole video. You should see 0 switches. This means that the tracker consistently maintained identities across the video.


```python
motmetrics = list(dict_vid_motmetrics.values())[0]
# motmetrics.loc[motmetrics['metric'] == 'num_switches']
motmetrics
```

Check global tracking accuracy. This represents the percentage of frames where the tracker correctly maintained identities for each instance. In this case, since there were no switches, the global tracking accuracy is 100% for all instances.


```python
gta = list(dict_vid_gta.values())[0]
gta
```
