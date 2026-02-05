"""DREEM Inference - Hugging Face Spaces App."""

import hashlib
from math import inf
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DREEM Tracking",
    page_icon="🔬",
    layout="wide",
)

# Constants
MAX_FRAMES_WARNING = 5000
MAX_FRAMES_LIMIT = 20000
SESSION_DIR = Path("/tmp/dreem_sessions")
MODEL_CACHE_DIR = Path("/tmp/dreem_models")

# Pretrained models available on HuggingFace Hub
PRETRAINED_MODELS = {
    "Animals (pretrained)": "talmolab/animals-pretrained",
    "Microscopy (pretrained)": "talmolab/microscopy-pretrained",
    "Upload custom checkpoint": None,
}


def get_timestamp() -> str:
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


@st.cache_resource
def download_pretrained_model(repo_id: str) -> str:
    """Download a pretrained model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'talmolab/dreem-animals-pretrained')

    Returns:
        Path to the downloaded checkpoint file
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Find the checkpoint file in the repo
    files = list_repo_files(repo_id)
    ckpt_files = [f for f in files if f.endswith(".ckpt")]

    if not ckpt_files:
        raise ValueError(f"No .ckpt file found in {repo_id}")

    # Download the first checkpoint found
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_files[0],
        cache_dir=MODEL_CACHE_DIR,
    )

    return ckpt_path


def get_session_dir() -> Path:
    """Get or create a session-specific directory."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:12]

    session_dir = SESSION_DIR / st.session_state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_video_info(video_path: str) -> dict:
    """Get video metadata (frame count, fps, duration)."""
    import imageio

    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()

    n_frames = meta.get("nframes", None)
    fps = meta.get("fps", 30)
    duration = meta.get("duration", None)

    if n_frames is None or n_frames == float("inf"):
        if duration and fps:
            n_frames = int(duration * fps)
        else:
            n_frames = reader.count_frames()

    reader.close()

    return {
        "n_frames": int(n_frames),
        "fps": fps,
        "duration": duration,
        "size_mb": os.path.getsize(video_path) / (1024 * 1024),
    }


def parse_slp_tracks(slp_path: str) -> dict:
    """Parse .slp file and extract track data per frame.

    Returns:
        Dict mapping frame_idx -> list of instances, where each instance is:
        {
            "track_id": int,
            "track_name": str,
            "keypoints": [(x, y, name), ...],  # all keypoints
            "centroid": (x, y),  # mean of valid keypoints
        }
    """
    import numpy as np
    import sleap_io as sio

    labels = sio.load_slp(slp_path)
    tracks_by_frame = {}

    for lf in labels.labeled_frames:
        frame_idx = lf.frame_idx
        instances = []

        for inst in lf.instances:
            if inst.track is None:
                continue
            track_name = inst.track.name

            # Get keypoints
            pts = inst.numpy()
            skeleton = labels.skeleton
            node_names = [n.name for n in skeleton.nodes] if skeleton else []

            keypoints = []
            valid_pts = []
            for i, pt in enumerate(pts):
                if not np.isnan(pt).any():
                    name = node_names[i] if i < len(node_names) else f"kp_{i}"
                    keypoints.append((float(pt[0]), float(pt[1]), name))
                    valid_pts.append(pt)

            # Compute centroid
            if valid_pts:
                centroid = np.mean(valid_pts, axis=0)
                centroid = (float(centroid[0]), float(centroid[1]))
            else:
                centroid = None

            if centroid:
                instances.append(
                    {
                        "track_id": track_name,
                        "keypoints": keypoints,
                        "centroid": centroid,
                    }
                )

        if instances:
            tracks_by_frame[frame_idx] = instances

    return tracks_by_frame


def get_frame(video_path: str, frame_idx: int):
    """Extract a single frame from video."""
    import imageio

    reader = imageio.get_reader(video_path, "ffmpeg")
    frame = reader.get_data(frame_idx)
    reader.close()
    return frame


def generate_annotated_video(
    video_path: str,
    tracks_data: dict,
    output_path: str,
    box_size: int = 64,
    show_keypoints: bool = True,
    show_labels: bool = True,
    fps: int = 30,
    progress_callback=None,
) -> str:
    """Generate an annotated MP4 video with track overlays.

    Args:
        video_path: Path to input video
        tracks_data: Dict mapping frame_idx -> list of instances
        output_path: Path to save output video
        box_size: Size of bounding boxes
        show_keypoints: Whether to draw keypoints
        show_labels: Whether to draw track labels
        fps: Output video frame rate
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Path to the generated video
    """
    import imageio

    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    n_frames = meta.get("nframes", 0)
    if n_frames == 0 or n_frames == float("inf"):
        # Count frames
        n_frames = reader.count_frames()

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")

    for frame_idx in range(n_frames):
        frame = reader.get_data(frame_idx)

        # Get instances for this frame and annotate
        instances = tracks_data.get(frame_idx, [])
        if instances:
            frame = annotate_frame(
                frame,
                instances,
                box_size=box_size,
                show_keypoints=show_keypoints,
                show_labels=show_labels,
            )

        writer.append_data(frame)

        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx + 1, n_frames)

    writer.close()
    reader.close()

    if progress_callback:
        progress_callback(n_frames, n_frames)

    return output_path


def annotate_frame(
    frame,
    instances: list,
    box_size: int = 64,
    show_keypoints: bool = True,
    show_labels: bool = True,
):
    """Overlay track annotations on a frame.

    Args:
        frame: numpy array (H, W, C)
        instances: list of instance dicts from parse_slp_tracks
        box_size: size of bounding box to draw
        show_keypoints: whether to draw keypoints
        show_labels: whether to draw track ID labels
    """
    import cv2
    import numpy as np

    frame = frame.copy()

    # Color palette (tab20)
    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ]

    for inst in instances:
        track_id = inst["track_id"]
        centroid = inst["centroid"]
        keypoints = inst["keypoints"]

        # Select color based on track_id
        color = colors[track_id % len(colors)]
        # Convert to BGR for cv2
        color_bgr = (color[2], color[1], color[0])

        cx, cy = int(centroid[0]), int(centroid[1])
        half = box_size // 2

        # Draw bounding box
        cv2.rectangle(
            frame,
            (cx - half, cy - half),
            (cx + half, cy + half),
            color_bgr,
            2,
        )

        # Draw keypoints
        if show_keypoints:
            for x, y, name in keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, color_bgr, -1)

        # Draw centroid
        cv2.circle(frame, (cx, cy), 6, color_bgr, -1)

        # Draw label
        if show_labels:
            label = f"Track {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (cx - half, cy - half - h - 10),
                (cx - half + w + 4, cy - half),
                color_bgr,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (cx - half + 2, cy - half - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    return frame


def create_progress_callback(batch_callback):
    """Create a Lightning callback that reports batch progress."""
    import pytorch_lightning as pl

    class StreamlitProgressCallback(pl.Callback):
        def __init__(self):
            self.total_batches = 0

        def on_predict_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            # Get total from the dataloader - it may be a single DataLoader or a list
            dataloaders = trainer.predict_dataloaders
            if isinstance(dataloaders, list):
                total = len(dataloaders[dataloader_idx])
            else:
                total = len(dataloaders)
            if batch_callback:
                batch_callback(batch_idx + 1, total)

    return StreamlitProgressCallback()


def run_inference(
    slp_path: str,
    video_path: str,
    checkpoint_path: str,
    output_dir: str,
    crop_size: int,
    anchor: str,
    clip_length: int,
    max_center_dist: float | None,
    max_detection_overlap: float | None,
    confidence_threshold: float,
    max_tracks: int | None,
    use_gpu: bool,
    progress_callback=None,
    batch_callback=None,
) -> str | None:
    """Run DREEM tracking inference with progress updates."""
    from omegaconf import OmegaConf

    from dreem.io import Config
    from dreem.models import GTRRunner

    def update_progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    input_dir = str(Path(slp_path).parent)

    cfg_dict = {
        "ckpt_path": checkpoint_path,
        "outdir": output_dir,
        "tracker": {
            "overlap_thresh": 0.01,
            "iou": "mult",
            "max_center_dist": max_center_dist,
            "mult_thresh": True,
            "confidence_threshold": confidence_threshold,
            "max_tracks": max_tracks,
        },
        "dataset": {
            "test_dataset": {
                "dir": {
                    "path": input_dir,
                    "labels_suffix": ".slp",
                    "vid_suffix": Path(video_path).suffix,
                },
                "slp_files": [slp_path],
                "video_files": [video_path],
                "anchors": anchor,
                "clip_length": clip_length,
                "crop_size": crop_size,
                "max_detection_overlap": max_detection_overlap,
                "chunk": True,
            }
        },
        "trainer": {
            "accelerator": "cpu",
        },
        "save_frame_meta": False,
    }

    cfg = OmegaConf.create(cfg_dict)

    try:
        # Stage 1: Load config
        update_progress(1, "Loading configuration...")
        pred_cfg = Config(cfg)

        # Stage 2: Load model
        update_progress(2, "Loading model checkpoint...")
        model = GTRRunner.load_from_checkpoint(checkpoint_path, strict=False)
        overrides_dict = model.setup_tracking(pred_cfg, mode="inference")

        # Stage 3: Setup data
        update_progress(3, "Setting up data pipeline...")
        labels_files, vid_files = pred_cfg.get_data_paths(
            "test", pred_cfg.cfg.dataset.test_dataset
        )

        # Create trainer with progress callback
        callbacks = []
        if batch_callback:
            callbacks.append(create_progress_callback(batch_callback))
        trainer = pred_cfg.get_trainer(callbacks=callbacks if callbacks else None)
        os.makedirs(output_dir, exist_ok=True)

        # Stage 4: Process each file
        update_progress(4, "Preparing inference...")
        output_path = None

        for i, (label_file, vid_file) in enumerate(zip(labels_files, vid_files)):
            dataset = pred_cfg.get_dataset(
                label_files=[label_file],
                vid_files=[vid_file],
                mode="test",
                overrides=overrides_dict,
            )
            dataloader = pred_cfg.get_dataloader(dataset, mode="test")

            # Run prediction - this is the long step
            update_progress(5, "Running tracker (this may take a while)...")
            preds = trainer.predict(model, dataloader)

            # Save results
            update_progress(6, "Saving results...")
            import sleap_io as sio
            from tqdm import tqdm

            from dreem.io.flags import FrameFlagCode

            pred_slp = []
            tracks = {}
            suggestions = []

            for batch in preds:
                for frame in batch:
                    if frame.frame_id.item() == 0:
                        video = (
                            sio.Video(frame.video)
                            if isinstance(frame.video, str)
                            else sio.Video
                        )
                    if frame.has_flag(FrameFlagCode.LOW_CONFIDENCE):
                        from sleap_io.model.suggestions import SuggestionFrame
                        suggestion = SuggestionFrame(
                            video=video, frame_idx=frame.frame_id.item()
                        )
                        suggestions.append(suggestion)
                    lf, tracks = frame.to_slp(tracks, video=video)
                    pred_slp.append(lf)

            pred_labels = sio.Labels(pred_slp, suggestions=suggestions)

            # Generate output filename
            if isinstance(vid_file, list):
                save_file_name = vid_file[0].split("/")[-2]
            else:
                save_file_name = vid_file

            timestamp = get_timestamp()
            output_path = os.path.join(
                output_dir,
                f"{Path(save_file_name).stem}.dreem_inference.{timestamp}.slp",
            )
            pred_labels.save(output_path)

        update_progress(7, "Complete!")
        return output_path

    except Exception as e:
        st.error(f"Inference failed: {e}")
        raise


def clear_session():
    """Clear session state and temp files."""
    import shutil

    if "session_id" in st.session_state:
        session_dir = SESSION_DIR / st.session_state.session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

    for key in list(st.session_state.keys()):
        del st.session_state[key]


# Initialize session state
if "inference_complete" not in st.session_state:
    st.session_state.inference_complete = False
if "tracks_data" not in st.session_state:
    st.session_state.tracks_data = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "output_slp_path" not in st.session_state:
    st.session_state.output_slp_path = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None


# App UI
st.title("DREEM: Multi-Object Tracking Across Biological Scales")
st.markdown(
    """
Upload your [SLEAP](https://sleap.ai) labels file (.slp) and video to run tracking. This demo only supports
input data in the .slp format, although the [CLI](https://dreem.sleap.ai/0.2.3/quickstart/) can be used with labeled masks in the [Cell Tracking Challenge](https://celltrackingchallenge.net/datasets/) format
as input for microscopy. For more information, see the [documentation](https://dreem.sleap.ai).
"""
)

# Sidebar for parameters
with st.sidebar:
    st.header("Inference Parameters")

    crop_size = st.number_input(
        "Crop Size",
        min_value=32,
        max_value=512,
        value=128,
        step=32,
        help="Size of bounding box to crop each instance (pixels)",
    )

    anchor = st.selectbox(
        "Anchor Point",
        options=["centroid", "thorax", "head", "abdomen"],
        index=0,
        help="Name of anchor keypoint for centering crops",
    )

    clip_length = st.number_input(
        "Clip Length",
        min_value=4,
        max_value=128,
        value=32,
        step=4,
        help="Number of frames per batch",
    )

    st.subheader("Tracker Settings")

    max_center_dist = st.number_input(
        "Max Center Distance",
        min_value=0,
        max_value=1000,
        value=0,
        help="Maximum distance between centers for association (0 = disabled)",
    )
    max_center_dist = max_center_dist if max_center_dist > 0 else None

    max_detection_overlap = st.slider(
        "Max Detection Overlap",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="IOU threshold above which detections are considered duplicates (0 = disabled)",
    )
    max_detection_overlap = max_detection_overlap if max_detection_overlap > 0 else None

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Threshold below which frames will be flagged as potential errors",
    )

    max_tracks = st.number_input(
        "Max Tracks",
        min_value=0,
        max_value=100,
        value=100,
        help="Maximum number of tracks to maintain (0 = unlimited)",
    )
    max_tracks = max_tracks if max_tracks > 0 else None

    use_gpu = st.checkbox(
        "Use GPU",
        value=False,
        help="Use GPU for inference (if available)",
    )

    st.subheader("Visualization")

    show_keypoints = st.checkbox("Show Keypoints", value=True)
    show_labels = st.checkbox("Show Track Labels", value=True)

    st.markdown("---")
    if st.button("Clear Session", type="secondary"):
        clear_session()
        st.rerun()


# Main content - show different UI based on state
if not st.session_state.inference_complete:
    # Upload and inference UI
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Files")

        slp_file = st.file_uploader(
            "SLEAP Labels File (.slp)",
            type=["slp"],
            help="Upload your SLEAP predictions file",
        )

        video_file = st.file_uploader(
            "Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload the corresponding video file (max 200MB)",
        )

        st.subheader("Model")
        model_choice = st.selectbox(
            "Select model",
            options=list(PRETRAINED_MODELS.keys()),
            index=0,
            help="Choose a pretrained model or upload your own",
        )

        # Show file uploader only if custom checkpoint is selected
        checkpoint_file = None
        if PRETRAINED_MODELS[model_choice] is None:
            checkpoint_file = st.file_uploader(
                "Model Checkpoint (.ckpt)",
                type=["ckpt"],
                help="Upload a trained DREEM model checkpoint",
            )

    with col2:
        st.subheader("Status")
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        progress_text_placeholder = st.empty()
        batch_text_placeholder = st.empty()

    # Run inference button
    if st.button("Run Tracking", type="primary", width="stretch"):
        # Validate inputs
        if not slp_file or not video_file:
            st.error("Please upload SLP and video files")
            st.stop()

        # Check checkpoint
        using_pretrained = PRETRAINED_MODELS[model_choice] is not None
        if not using_pretrained and not checkpoint_file:
            st.error("Please upload a model checkpoint or select a pretrained model")
            st.stop()

        session_dir = get_session_dir()
        output_dir = session_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Save uploaded files
        slp_path = session_dir / slp_file.name
        video_path = session_dir / video_file.name

        # Get checkpoint path (download pretrained or use uploaded)
        if using_pretrained:
            status_placeholder.info(f"Downloading {model_choice} model...")
            try:
                ckpt_path = download_pretrained_model(PRETRAINED_MODELS[model_choice])
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
        else:
            ckpt_path = session_dir / checkpoint_file.name
            with open(ckpt_path, "wb") as f:
                f.write(checkpoint_file.getvalue())

        # Save uploaded data files
        with open(slp_path, "wb") as f:
            f.write(slp_file.getvalue())
        with open(video_path, "wb") as f:
            f.write(video_file.getvalue())

        # Validate video
        status_placeholder.info("Validating video...")
        try:
            video_info = get_video_info(str(video_path))
            n_frames = video_info["n_frames"]

            st.info(
                f"Video: {n_frames:,} frames, "
                f"{video_info['fps']:.1f} fps, "
                f"{video_info['size_mb']:.1f} MB"
            )

            if n_frames > MAX_FRAMES_LIMIT:
                st.error(
                    f"Video has {n_frames:,} frames, which exceeds the limit of "
                    f"{MAX_FRAMES_LIMIT:,}. Please upload a shorter video."
                )
                st.stop()
            elif n_frames > MAX_FRAMES_WARNING:
                st.warning(
                    f"Video has {n_frames:,} frames. Processing may be slow."
                )

            st.session_state.video_info = video_info

        except Exception as e:
            st.warning(f"Could not validate video: {e}. Proceeding anyway...")
            st.session_state.video_info = {"n_frames": 1000, "fps": 30}

        # Run inference with progress indicator
        status_placeholder.empty()
        progress_bar = progress_placeholder.progress(0)
        progress_text = progress_text_placeholder
        batch_text = batch_text_placeholder

        def update_progress(stage: int, msg: str):
            # Map stages to progress (stages 1-4 are setup, 5 is inference, 6-7 are saving)
            if stage <= 4:
                progress_bar.progress(stage * 0.1)  # 0-40%
            elif stage == 5:
                progress_bar.progress(0.4)  # Will be updated by batch callback
            elif stage >= 6:
                progress_bar.progress(0.9 + (stage - 6) * 0.05)  # 90-100%
            progress_text.text(msg)

        def update_batch_progress(current: int, total: int):
            # During inference (stage 5), show batch progress from 40% to 90%
            pct = 0.4 + (current / total) * 0.5
            progress_bar.progress(min(pct, 0.9))
            batch_text.text(f"Batch {current}/{total}")

        try:
            output_slp = run_inference(
                slp_path=str(slp_path),
                video_path=str(video_path),
                checkpoint_path=str(ckpt_path),
                output_dir=str(output_dir),
                crop_size=crop_size,
                anchor=anchor,
                clip_length=clip_length,
                max_center_dist=max_center_dist,
                max_detection_overlap=max_detection_overlap,
                confidence_threshold=confidence_threshold,
                max_tracks=max_tracks,
                use_gpu=use_gpu,
                progress_callback=update_progress,
                batch_callback=update_batch_progress,
            )

            # Clear progress indicators
            progress_placeholder.empty()
            progress_text_placeholder.empty()
            batch_text_placeholder.empty()

            if output_slp:
                status_placeholder.success("Inference complete!")

                # Parse tracks and store in session state
                status_placeholder.info("Parsing tracks...")
                tracks_data = parse_slp_tracks(output_slp)

                st.session_state.tracks_data = tracks_data
                st.session_state.video_path = str(video_path)
                st.session_state.output_slp_path = output_slp
                st.session_state.inference_complete = True

                st.rerun()
            else:
                status_placeholder.error("Inference produced no output")

        except Exception as e:
            status_placeholder.error(f"Error: {e}")
            st.exception(e)

else:
    # Visualization UI - inference is complete
    st.success("Tracking complete!")

    tracks_data = st.session_state.tracks_data
    video_path = st.session_state.video_path
    video_info = st.session_state.video_info

    # Generate annotated video if not already done
    if "annotated_video_path" not in st.session_state or st.session_state.annotated_video_path is None:
        st.info("Generating annotated video...")
        video_progress = st.progress(0)
        video_status = st.empty()

        def video_progress_callback(current, total):
            video_progress.progress(current / total)
            video_status.text(f"Processing frame {current}/{total}")

        session_dir = get_session_dir()
        annotated_video_path = str(session_dir / "tracked_video.mp4")

        try:
            generate_annotated_video(
                video_path=video_path,
                tracks_data=tracks_data,
                output_path=annotated_video_path,
                box_size=crop_size,
                show_keypoints=show_keypoints,
                show_labels=show_labels,
                fps=int(video_info.get("fps", 30)),
                progress_callback=video_progress_callback,
            )
            st.session_state.annotated_video_path = annotated_video_path
            video_progress.empty()
            video_status.empty()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate video: {e}")
            video_progress.empty()
            video_status.empty()
    else:
        # Video already generated - display it
        annotated_video_path = st.session_state.annotated_video_path

        # Download buttons
        st.subheader("Downloads")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.output_slp_path:
                with open(st.session_state.output_slp_path, "rb") as f:
                    st.download_button(
                        label="Download Tracked Labels (.slp)",
                        data=f.read(),
                        file_name="tracked_output.slp",
                        mime="application/octet-stream",
                    )
        with col2:
            if os.path.exists(annotated_video_path):
                with open(annotated_video_path, "rb") as f:
                    st.download_button(
                        label="Download Annotated Video (.mp4)",
                        data=f.read(),
                        file_name="tracked_video.mp4",
                        mime="video/mp4",
                    )

        # Display video (constrained width)
        st.subheader("Annotated Video")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(annotated_video_path)

        # Stats
        with st.expander("Track Statistics"):
            frames_with_tracks = sorted(tracks_data.keys())
            all_track_ids = set()
            for instances in tracks_data.values():
                for inst in instances:
                    all_track_ids.add(inst["track_id"])

            st.write(f"**Total unique tracks:** {len(all_track_ids)}")
            st.write(f"**Frames with tracks:** {len(frames_with_tracks)}")
            st.write(f"**Track IDs:** {sorted(all_track_ids)}")


# Footer
st.markdown("---")
st.markdown(
    """
[GitHub](https://github.com/talmolab/dreem) |
[Documentation](https://dreem.sleap.ai)
"""
)
