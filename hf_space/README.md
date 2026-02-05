---
title: DREEM Tracking
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: bsd-3-clause
---

# DREEM: Multi-Object Tracking

This Space runs **DREEM** (Global Tracking Transformer) inference for biological multi-object tracking.

## Usage

1. **Upload your SLEAP labels file** (.slp) containing detections/poses
2. **Upload the corresponding video file** (.mp4, .avi, .mov, .mkv)
3. **Upload a trained DREEM checkpoint** (.ckpt)
4. **Adjust parameters** in the sidebar as needed
5. **Click "Run Tracking"** to run inference

## Parameters

### Inference Parameters

- **Crop Size**: Size of bounding box to crop each instance (pixels)
- **Anchor Point**: Keypoint to use for centering crops (e.g., "centroid")
- **Clip Length**: Number of frames processed per batch

### Tracker Settings

- **Max Center Distance**: Maximum distance between centers for association
- **Max Detection Overlap**: IOU threshold above which detections are considered duplicates
- **Confidence Threshold**: Threshold below which frames will be flagged as potential errors
- **Max Tracks**: Maximum number of tracks to maintain

### Visualization

- **Box Size**: Size of bounding boxes in frame viewer
- **Show Keypoints**: Toggle keypoint display
- **Show Track Labels**: Toggle track ID labels

## Output

- **Tracked Labels (.slp)**: SLEAP labels file with track assignments
- **Interactive Frame Viewer**: Browse frames with a slider, view annotations overlaid on video frames
  - Keypoints with colored markers
  - Bounding boxes around each instance
  - Track ID labels
  - Navigation buttons to jump between frames with tracks
  - Track statistics panel

## Links

- [GitHub Repository](https://github.com/talmolab/dreem)
- [Documentation](https://talmolab.github.io/dreem/)
- [Paper](https://arxiv.org/abs/2312.06539)

## Citation

```bibtex
@article{dreem2023,
  title={DREEM: Global Tracking Transformers for Multi-Object Tracking},
  author={Sheridan, Arlo and Prasad, Aaditya and Tu, Vincent and Manor, Uri and Pereira, Talmo},
  year={2023}
}
```
