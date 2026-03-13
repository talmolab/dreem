"""Helper functions for visualizing tracking."""

from __future__ import annotations

import logging
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import hydra
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from tqdm import tqdm

if TYPE_CHECKING:
    import sleap_io as sio

logger = logging.getLogger("dreem.io")

palette = sns.color_palette("tab20")


def fill_missing(data: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Fill missing values independently along each dimension after the first.

    Args:
        data: the array for which to fill missing value
        kind: How to interpolate missing values using `scipy.interpolate.interp1d`

    Returns:
        The array with missing values filled in
    """
    # Store initial shape.
    initial_shape = data.shape

    # Flatten after first dim.
    data = data.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(data.shape[-1]):
        y = data[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        data[:, i] = y

    # Restore to initial shape.
    data = data.reshape(initial_shape)

    return data


def annotate_video(
    video: "imageio.core.format.Reader",
    labels: pd.DataFrame,
    key: str,
    color_palette: list | str = palette,
    trails: int = 2,
    boxes: int = (64, 64),
    names: bool = True,
    track_scores=0.5,
    centroids: int = 4,
    poses: bool = False,
    save_path: str = "debug_animal.mp4",
    fps: int = 30,
    alpha: float = 0.2,
) -> list:
    """Annotate video frames with labels.

    Labels video with bboxes, centroids, trajectory trails, and/or poses.

    Args:
        video: The video to be annotated in an ndarray
        labels: The pandas dataframe containing the centroid and/or pose locations of the instances
        key: The key where labels are stored in the dataframe - mostly used for choosing whether to annotate based on pred or gt labels
        color_palette: The matplotlib colorpalette to use for annotating the video. Defaults to `tab10`
        trails: The size of the trajectory trail. If trails size <= 0 or None then it is not added
        boxes: The size of the bbox. If bbox size <= 0 or None then it is not added
        names: Whether or not to annotate with name
        centroids: The size of the centroid. If centroid size <= 0 or None then it is not added
        poses: Whether or not to annotate with poses
        save_path: The path to save the annotated video.
        fps: The frame rate of the generated video
        track_scores: Minimum track score threshold for displaying tracks
        alpha: The opacity of the annotations.

    Returns:
        A list of annotated video frames
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required for video annotation. "
            "Install with: pip install opencv-python"
        ) from e

    writer = imageio.get_writer(save_path, fps=fps)
    color_palette = (
        sns.color_palette(color_palette)
        if isinstance(color_palette, str)
        else deepcopy(color_palette)
    )

    if trails:
        track_trails = {}
    try:
        for i in tqdm(sorted(labels["Frame"].unique()), desc="Frame", unit="Frame"):
            frame = video.get_data(i)
            if frame.shape[0] == 1 or frame.shape[-1] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            # else:
            #     frame = frame.copy()

            lf = labels[labels["Frame"] == i]
            for idx, instance in lf.iterrows():
                if not trails:
                    track_trails = {}

                if poses:
                    # TODO figure out best way to store poses (maybe pass a slp labels file too?)
                    trails = False
                    centroids = False
                    for idx, (pose, edge) in enumerate(
                        zip(instance["poses"], instance["edges"])
                    ):
                        pose = fill_missing(pose.numpy())

                        pred_track_id = instance[key][idx].numpy().tolist()

                        # Add midpt to track trail.
                        if pred_track_id not in list(track_trails.keys()):
                            track_trails[pred_track_id] = []

                        # Select a color based on track_id.
                        track_color_idx = pred_track_id % len(color_palette)
                        track_color = (
                            (np.array(color_palette[track_color_idx]) * 255)
                            .astype(np.uint8)
                            .tolist()[::-1]
                        )

                        for p in pose:
                            # try:
                            #    p = tuple([int(i) for i in p.numpy()][::-1])
                            # except:
                            #    continue

                            p = tuple(int(i) for i in p)[::-1]

                            track_trails[pred_track_id].append(p)

                            frame = cv2.circle(
                                frame, p, radius=2, color=track_color, thickness=-1
                            )

                        for e in edge:
                            source = tuple(int(i) for i in pose[int(e[0])])[::-1]
                            target = tuple(int(i) for i in pose[int(e[1])])[::-1]

                            frame = cv2.line(frame, source, target, track_color, 1)

                if (boxes) or centroids:
                    # Get coordinates for detected objects in the current frame.
                    if isinstance(boxes, int):
                        boxes = (boxes, boxes)

                    box_w, box_h = boxes
                    x = instance["X"]
                    y = instance["Y"]
                    min_x, min_y, max_x, max_y = (
                        int(x - box_w / 2),
                        int(y - box_h / 2),
                        int(x + box_w / 2),
                        int(y + box_h / 2),
                    )
                    midpt = (int(x), int(y))

                    pred_track_id = instance[key]

                    if "Track_score" in instance.index:
                        track_score = instance["Track_score"]
                    else:
                        track_scores = 0

                    # Add midpt to track trail.
                    if pred_track_id not in list(track_trails.keys()):
                        track_trails[pred_track_id] = []
                    track_trails[pred_track_id].append(midpt)

                    # Select a color based on track_id.
                    track_color_idx = int(pred_track_id) % len(color_palette)
                    track_color = (
                        (np.array(color_palette[track_color_idx]) * 255)
                        .astype(np.uint8)
                        .tolist()[::-1]
                    )

                    # Bbox.
                    if boxes is not None:
                        frame = cv2.rectangle(
                            frame,
                            (min_x, min_y),
                            (max_x, max_y),
                            color=track_color,
                            thickness=2,
                        )

                    # Track trail.
                    if centroids:
                        frame = cv2.circle(
                            frame,
                            midpt,
                            radius=centroids,
                            color=track_color,
                            thickness=-1,
                        )
                        for i in range(0, len(track_trails[pred_track_id]) - 1):
                            frame = cv2.addWeighted(
                                cv2.circle(
                                    frame,  # .copy(),
                                    track_trails[pred_track_id][i],
                                    radius=4,
                                    color=track_color,
                                    thickness=-1,
                                ),
                                alpha,
                                frame,
                                1 - alpha,
                                0,
                            )
                            if trails:
                                frame = cv2.line(
                                    frame,
                                    track_trails[pred_track_id][i],
                                    track_trails[pred_track_id][i + 1],
                                    color=track_color,
                                    thickness=trails,
                                )

                # Track name.
                name_str = ""

                if names:
                    name_str += f"track_{pred_track_id}"
                if names and track_scores:
                    name_str += " | "
                if track_scores:
                    name_str += f"score: {track_score:0.3f}"

                if len(name_str) > 0:
                    frame = cv2.putText(
                        frame,
                        # f"idx:{idx} | track_{pred_track_id}",
                        name_str,
                        org=(int(min_x), max(0, int(min_y) - 10)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9,
                        color=track_color,
                        thickness=2,
                    )
            writer.append_data(frame)
            # if i % fps == 0:
            #     gc.collect()

    except Exception as e:
        writer.close()
        logger.exception(e)
        return False

    writer.close()
    return True


def save_vid(
    annotated_frames: list,
    save_path: str = "debug_animal",
    fps: int = 30,
):
    """Save video to file.

    Args:
        annotated_frames: a list of frames annotated by `annotate_frames`
        save_path: The path of the annotated file.
        fps: The frame rate in frames per second of the annotated video
    """
    for idx, (ds_name, data) in enumerate([(save_path, annotated_frames)]):
        imageio.mimwrite(f"{ds_name}.mp4", data, fps=fps, macro_block_size=1)


def color(val: float, thresh: float = 0.01) -> str:
    """Highlight value in dataframe if it is over a threshold.

    Args:
        val: The value to color
        thresh: The threshold for which to color

    Returns:
        A string containing how to highlight the value
    """
    color = "lightblue" if float(val) > thresh else ""
    return f"background-color: {color}"


def bold(val: float, thresh: float = 0.01) -> str:
    """Bold value if it is over a threshold.

    Args:
        val: The value to bold or not
        thresh: The threshold the value has to exceed to be bolded

    Returns:
        A string indicating how to bold the item.
    """
    bold = "bold" if float(val) > thresh else ""
    return f"font-weight: {bold}"


def extract_centroids_from_masks(
    masks: np.ndarray,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Extract centroid (x, y) for each track ID in each frame from a mask stack.

    Args:
        masks: A (T, H, W) uint16 array where pixel values are track IDs
            (0 = background).

    Returns:
        Nested dict: ``centroids[frame_idx][track_id] = (cx, cy)`` where
        cx, cy are in pixel coordinates.
    """
    from scipy.ndimage import center_of_mass

    centroids: dict[int, dict[int, tuple[float, float]]] = {}
    for t in range(masks.shape[0]):
        frame_mask = masks[t]
        track_ids = np.unique(frame_mask)
        track_ids = track_ids[track_ids != 0]
        if len(track_ids) == 0:
            centroids[t] = {}
            continue
        # center_of_mass returns (row, col) per label
        coms = center_of_mass(frame_mask, labels=frame_mask, index=track_ids)
        frame_centroids: dict[int, tuple[float, float]] = {}
        for tid, (row, col) in zip(track_ids, coms):
            frame_centroids[int(tid)] = (float(col), float(row))  # (x, y)
        centroids[t] = frame_centroids
    return centroids


def masks_to_sleap_labels(
    masks: np.ndarray,
    video: sio.Video | None = None,
    centroids: dict[int, dict[int, tuple[float, float]]] | None = None,
) -> sio.Labels:
    """Convert a CTC mask stack to a sleap-io Labels object.

    Creates a single-node skeleton with a "centroid" node. For each frame and
    track ID, creates a ``SegmentationMask`` from the binary mask and an
    ``Instance`` positioned at the centroid.

    Args:
        masks: A (T, H, W) uint16 array where pixel values are track IDs.
        video: Optional sleap-io Video to associate with the labels.
        centroids: Precomputed centroids dict from
            ``extract_centroids_from_masks()``. If None, computed from masks.

    Returns:
        A sleap-io Labels object with masks, instances, and tracks.
    """
    import sleap_io as sio
    from sleap_io.model.mask import SegmentationMask

    if centroids is None:
        centroids = extract_centroids_from_masks(masks)

    skeleton = sio.Skeleton(nodes=["centroid"])
    tracks: dict[int, sio.Track] = {}
    labeled_frames: list[sio.LabeledFrame] = []
    all_masks: list[SegmentationMask] = []

    for t in range(masks.shape[0]):
        frame_mask = masks[t]
        frame_centroids = centroids.get(t, {})
        instances: list[sio.Instance] = []

        for tid, (cx, cy) in frame_centroids.items():
            if tid not in tracks:
                tracks[tid] = sio.Track(name=str(tid))
            track = tracks[tid]

            # Create binary mask for this track
            binary_mask = (frame_mask == tid).astype(np.uint8)
            seg_mask = SegmentationMask.from_numpy(
                binary_mask,
                video=video,
                frame_idx=t,
                track=track,
            )
            all_masks.append(seg_mask)

            # Create instance at centroid
            instance = sio.Instance.from_numpy(
                np.array([[cx, cy]]),
                skeleton=skeleton,
                track=track,
            )
            instances.append(instance)

        lf = sio.LabeledFrame(
            video=video if video is not None else sio.Video(""),
            frame_idx=t,
            instances=instances,
        )
        labeled_frames.append(lf)

    labels = sio.Labels(
        labeled_frames=labeled_frames,
        skeletons=[skeleton],
        tracks=list(tracks.values()),
        masks=all_masks,
    )
    return labels


def _extract_centroids_from_labels(
    labels: sio.Labels,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Extract centroid (x, y) for each tracked instance in each frame from Labels.

    For each instance with a track assignment, computes the centroid as the mean
    of all valid (non-NaN) keypoint coordinates.

    Args:
        labels: A sleap-io Labels object with tracked instances.

    Returns:
        Nested dict: ``centroids[frame_idx][track_id] = (cx, cy)`` where
        cx, cy are in pixel coordinates and track_id is the integer index of the
        track in ``labels.tracks``.
    """
    track_to_id = {track: i for i, track in enumerate(labels.tracks)}
    centroids: dict[int, dict[int, tuple[float, float]]] = {}

    for lf in labels.labeled_frames:
        frame_centroids: dict[int, tuple[float, float]] = {}
        for inst in lf.instances:
            if inst.track is None:
                continue
            tid = track_to_id.get(inst.track)
            if tid is None:
                continue
            pts = inst.numpy()  # (n_nodes, 2)
            valid = ~np.isnan(pts).any(axis=1)
            if not valid.any():
                continue
            mean_pt = pts[valid].mean(axis=0)
            frame_centroids[tid] = (float(mean_pt[0]), float(mean_pt[1]))
        centroids[lf.frame_idx] = frame_centroids

    return centroids


def render_slp_video(
    labels: sio.Labels | str | Path,
    save_path: str | Path,
    *,
    palette: str = "distinct",
    marker_size: float = 4.0,
    line_width: float = 2.0,
    trail_length: int = 10,
    trail_width: float = 1.5,
    show_ids: bool = True,
    show_trails: bool = True,
    show_nodes: bool = True,
    show_edges: bool = True,
    id_font_size: float = 12.0,
    fps: float | None = None,
    scale: float = 1.0,
    crf: int = 25,
    show_progress: bool = True,
) -> Path:
    """Render a SLEAP labels file as a video with colored tracks.

    Wraps sleap-io's ``render_video()`` with trajectory trails and ID labels
    using the same callback factories as ``render_ctc_video()``.

    Args:
        labels: A sleap-io Labels object or path to a ``.slp`` file.
        save_path: Output video file path (e.g., "output.mp4").
        palette: Color palette name (passed to sleap-io's ``get_palette``).
        marker_size: Size of keypoint markers in pixels.
        line_width: Width of skeleton edge lines in pixels.
        trail_length: Number of past frames to include in trajectory trails.
        trail_width: Width of trail lines in pixels.
        show_ids: Whether to draw track ID labels.
        show_trails: Whether to draw trajectory trails.
        show_nodes: Whether to draw keypoint markers.
        show_edges: Whether to draw skeleton edges.
        id_font_size: Font size for ID labels in pixels.
        fps: Output video frame rate. If None, uses the source video's FPS.
        scale: Scale factor for rendering (e.g., 2.0 for 2x).
        crf: Constant Rate Factor for H.264 encoding (lower = higher quality).
        show_progress: Whether to show a progress bar.

    Returns:
        Path to the output video file.
    """
    import sleap_io as sio
    from sleap_io.rendering.colors import get_palette

    save_path = Path(save_path)

    # Load labels from path if needed
    if isinstance(labels, (str, Path)):
        labels = sio.load_slp(str(labels))

    # Extract centroids for trail callback
    centroids = _extract_centroids_from_labels(labels)
    n_tracks = len(labels.tracks)

    # Build color map: track index -> (r, g, b)
    palette_colors = get_palette(palette, max(n_tracks, 1))
    track_id_to_color: dict[int, tuple[int, int, int]] = {}
    for i in range(n_tracks):
        track_id_to_color[i] = palette_colors[i % len(palette_colors)]

    # Build callbacks
    trail_cb = None
    if show_trails:
        trail_cb = _make_trail_callback(
            centroids, track_id_to_color, trail_length, trail_width
        )

    id_cb = None
    if show_ids:
        id_cb = _make_id_label_callback(track_id_to_color, id_font_size)

    # Build render kwargs
    render_kwargs: dict = dict(
        save_path=save_path,
        color_by="track",
        palette=palette,
        marker_size=marker_size,
        line_width=line_width,
        show_nodes=show_nodes,
        show_edges=show_edges,
        scale=scale,
        crf=crf,
        show_progress=show_progress,
        post_render_callback=trail_cb,
        per_instance_callback=id_cb,
    )
    if fps is not None:
        render_kwargs["fps"] = fps

    sio.render_video(labels, **render_kwargs)

    return save_path


def _make_trail_callback(
    centroids: dict[int, dict[int, tuple[float, float]]],
    colors: dict[int, tuple[int, int, int]],
    trail_length: int = 10,
    trail_width: float = 1.5,
) -> Callable:
    """Create a post_render_callback that draws fading trajectory trails.

    Args:
        centroids: Nested dict ``centroids[frame_idx][track_id] = (cx, cy)``.
        colors: Dict ``colors[track_id] = (r, g, b)`` for each track.
        trail_length: Number of past frames to include in the trail.
        trail_width: Width of trail lines in pixels.

    Returns:
        A callback function compatible with ``render_frame``'s
        ``post_render_callback`` parameter.
    """
    import skia

    trail_history: dict[int, deque[tuple[float, float]]] = {}

    def callback(ctx) -> None:
        frame_idx = ctx.frame_idx
        frame_centroids = centroids.get(frame_idx, {})

        # Update trail history
        for tid, (cx, cy) in frame_centroids.items():
            if tid not in trail_history:
                trail_history[tid] = deque(maxlen=trail_length)
            canvas_pt = ctx.world_to_canvas(cx, cy)
            trail_history[tid].append(canvas_pt)

        # Draw trails
        for tid, points in trail_history.items():
            if len(points) < 2:
                continue
            r, g, b = colors.get(tid, (255, 255, 255))
            n_pts = len(points)
            for i in range(n_pts - 1):
                # Fade alpha from old to new
                alpha = int(255 * (i + 1) / n_pts)
                paint = skia.Paint(
                    Color=skia.Color(r, g, b, alpha),
                    StrokeWidth=trail_width * ctx.scale,
                    Style=skia.Paint.kStroke_Style,
                    AntiAlias=True,
                )
                ctx.canvas.drawLine(
                    points[i][0],
                    points[i][1],
                    points[i + 1][0],
                    points[i + 1][1],
                    paint,
                )

    return callback


def _make_id_label_callback(
    colors: dict[int, tuple[int, int, int]],
    font_size: float = 12.0,
) -> Callable:
    """Create a per_instance_callback that draws track ID labels.

    Draws a colored rounded-rect background pill with white text showing
    the track ID near each instance's centroid.

    Args:
        colors: Dict ``colors[track_id] = (r, g, b)`` for each track.
        font_size: Font size in pixels.

    Returns:
        A callback function compatible with ``render_frame``'s
        ``per_instance_callback`` parameter.
    """
    import skia

    def callback(ctx) -> None:
        if ctx.track_id is None:
            return
        centroid = ctx.get_centroid()
        if centroid is None:
            return

        cx, cy = ctx.world_to_canvas(centroid[0], centroid[1])
        label = str(ctx.track_id)
        r, g, b = colors.get(ctx.track_id, (255, 255, 255))

        scaled_font_size = font_size * ctx.scale

        # Create font and measure text
        font = skia.Font(None, scaled_font_size)
        text_width = font.measureText(label)
        text_height = scaled_font_size

        # Position label above centroid
        pad_x, pad_y = 4 * ctx.scale, 2 * ctx.scale
        label_x = cx - text_width / 2
        label_y = cy - text_height - 6 * ctx.scale

        # Draw background pill
        bg_rect = skia.Rect.MakeXYWH(
            label_x - pad_x,
            label_y - pad_y,
            text_width + 2 * pad_x,
            text_height + 2 * pad_y,
        )
        bg_paint = skia.Paint(
            Color=skia.Color(r, g, b, 200),
            Style=skia.Paint.kFill_Style,
            AntiAlias=True,
        )
        ctx.canvas.drawRoundRect(bg_rect, 4 * ctx.scale, 4 * ctx.scale, bg_paint)

        # Draw text
        text_paint = skia.Paint(
            Color=skia.Color(255, 255, 255, 255),
            AntiAlias=True,
        )
        ctx.canvas.drawString(label, label_x, label_y + text_height, font, text_paint)

    return callback


def _blend_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a colored mask overlay onto a frame.

    Args:
        frame: RGB uint8 array of shape (H, W, 3).
        mask: Binary mask of shape (H, W) (bool or uint8).
        color: RGB color tuple for the mask.
        alpha: Opacity of the mask overlay (0.0 to 1.0).

    Returns:
        Frame with mask overlay blended in.
    """
    result = frame.copy()
    mask_bool = mask.astype(bool)
    overlay = np.array(color, dtype=np.float32)
    result[mask_bool] = (
        (1 - alpha) * result[mask_bool].astype(np.float32) + alpha * overlay
    ).astype(np.uint8)
    return result


def render_ctc_video(
    masks: np.ndarray | str | Path,
    save_path: str | Path,
    raw_frames: np.ndarray | str | Path | None = None,
    *,
    mask_alpha: float = 0.5,
    palette: str = "distinct",
    marker_size: float = 4.0,
    trail_length: int = 10,
    trail_width: float = 1.5,
    show_ids: bool = True,
    show_masks: bool = True,
    show_trails: bool = True,
    show_centroids: bool = True,
    id_font_size: float = 12.0,
    fps: float = 30.0,
    scale: float = 1.0,
    crf: int = 25,
    show_progress: bool = True,
) -> Path:
    """Render a CTC mask tracking result as a video with colored overlays.

    Takes a uint16 mask stack (pixel values = track IDs) and renders a video
    with colored mask overlays, centroid markers, trajectory trails, and
    ID labels using sleap-io's skia-based rendering pipeline.

    Args:
        masks: A (T, H, W) uint16 array or path to a TIFF stack where pixel
            values are track IDs (0 = background).
        save_path: Output video file path (e.g., "output.mp4").
        raw_frames: Optional raw video frames as a (T, H, W) or (T, H, W, C)
            array, or path to a TIFF/video file. If None, uses a black
            background.
        mask_alpha: Opacity of mask overlays (0.0 to 1.0).
        palette: Color palette name (passed to sleap-io's ``get_palette``).
        marker_size: Size of centroid markers in pixels.
        trail_length: Number of past frames to include in trajectory trails.
        trail_width: Width of trail lines in pixels.
        show_ids: Whether to draw track ID labels.
        show_masks: Whether to draw colored mask overlays.
        show_trails: Whether to draw trajectory trails.
        show_centroids: Whether to draw centroid markers.
        id_font_size: Font size for ID labels in pixels.
        fps: Output video frame rate.
        scale: Scale factor for rendering (e.g., 2.0 for 2x).
        crf: Constant Rate Factor for H.264 encoding (lower = higher quality).
        show_progress: Whether to show a progress bar.

    Returns:
        Path to the output video file.
    """
    import tifffile
    from sleap_io.rendering.colors import get_palette
    from sleap_io.rendering.core import render_frame

    save_path = Path(save_path)

    # Load masks
    if isinstance(masks, (str, Path)):
        masks = tifffile.imread(str(masks)).astype(np.uint16)
    if masks.ndim != 3:
        raise ValueError(f"Expected 3D mask array (T, H, W), got shape {masks.shape}")

    n_frames, height, width = masks.shape

    # Load raw frames
    frames_arr: np.ndarray | None = None
    if raw_frames is not None:
        if isinstance(raw_frames, (str, Path)):
            raw_path = Path(raw_frames)
            if raw_path.suffix.lower() in (".tif", ".tiff"):
                frames_arr = tifffile.imread(str(raw_path))
            else:
                reader = imageio.get_reader(str(raw_path))
                frames_arr = np.stack([reader.get_data(i) for i in range(len(reader))])
                reader.close()
        else:
            frames_arr = raw_frames

    # Extract centroids and track IDs
    centroids = extract_centroids_from_masks(masks)
    all_track_ids = sorted(set(tid for fc in centroids.values() for tid in fc.keys()))
    n_tracks = len(all_track_ids)

    # Build color map
    palette_colors = get_palette(palette, max(n_tracks, 1))
    track_id_to_color: dict[int, tuple[int, int, int]] = {}
    for i, tid in enumerate(all_track_ids):
        track_id_to_color[tid] = palette_colors[i % len(palette_colors)]

    # Map track IDs to sequential indices for render_frame
    track_id_to_idx: dict[int, int] = {tid: i for i, tid in enumerate(all_track_ids)}

    # Build callbacks
    trail_cb = None
    if show_trails:
        trail_cb = _make_trail_callback(
            centroids, track_id_to_color, trail_length, trail_width
        )

    id_cb = None
    if show_ids:
        id_cb = _make_id_label_callback(track_id_to_color, id_font_size)

    # Set up video writer
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(save_path),
        fps=fps,
        codec="libx264",
        output_params=["-crf", str(crf)],
    )

    iterator = range(n_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Rendering", unit="frame")

    try:
        for t in iterator:
            # Get base frame
            if frames_arr is not None:
                frame = frames_arr[t]
            else:
                frame = np.zeros((height, width), dtype=np.uint8)

            # Ensure RGB uint8
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[0] in (1, 3):
                # Channel-first (C, H, W) -> (H, W, C)
                frame = np.moveaxis(frame, 0, -1)
            if frame.shape[-1] == 1:
                frame = np.concatenate([frame] * 3, axis=-1)

            # Normalize to uint8 if needed
            if frame.dtype != np.uint8:
                if frame.max() > 255:
                    frame = ((frame.astype(np.float32) / frame.max()) * 255).astype(
                        np.uint8
                    )
                else:
                    frame = frame.astype(np.uint8)

            # Draw mask overlays
            frame_centroids = centroids.get(t, {})
            if show_masks:
                for tid in frame_centroids:
                    binary_mask = masks[t] == tid
                    color = track_id_to_color.get(tid, (255, 255, 255))
                    frame = _blend_mask_overlay(frame, binary_mask, color, mask_alpha)

            # Build instance data for render_frame
            instances_points: list[np.ndarray] = []
            track_indices: list[int] = []
            instance_metadata: list[dict] = []
            for tid, (cx, cy) in frame_centroids.items():
                instances_points.append(np.array([[cx, cy]]))
                track_indices.append(track_id_to_idx[tid])
                instance_metadata.append(
                    {
                        "track_id": tid,
                        "track_name": str(tid),
                    }
                )

            if show_centroids and instances_points:
                rendered = render_frame(
                    frame,
                    instances_points,
                    edge_inds=[],
                    node_names=["centroid"],
                    color_by="track",
                    palette=palette,
                    marker_size=marker_size,
                    show_edges=False,
                    scale=scale,
                    track_indices=track_indices,
                    n_tracks=n_tracks,
                    post_render_callback=trail_cb,
                    per_instance_callback=id_cb,
                    frame_idx=t,
                    instance_metadata=instance_metadata,
                )
            else:
                # No centroids to render, but may still need trails/ids
                # Render with empty instances to trigger callbacks
                rendered = render_frame(
                    frame,
                    [],
                    edge_inds=[],
                    node_names=["centroid"],
                    color_by="track",
                    palette=palette,
                    marker_size=marker_size,
                    show_edges=False,
                    scale=scale,
                    track_indices=[],
                    n_tracks=n_tracks,
                    post_render_callback=trail_cb,
                    frame_idx=t,
                )

            # Convert RGBA to RGB for video
            if rendered.shape[-1] == 4:
                rendered = rendered[:, :, :3]

            writer.append_data(rendered)
    finally:
        writer.close()

    return save_path


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    """Take in a path to a video + labels file, annotates a video and saves it to the specified path."""
    labels = pd.read_csv(cfg.labels_path)
    video = imageio.get_reader(cfg.vid_path, "ffmpeg")
    frames_annotated = annotate_video(
        video, labels, save_path=cfg.save_path, **cfg.annotate
    )

    if frames_annotated:
        logger.info("Video saved to {cfg.save_path}!")
    else:
        logger.error("Failed to annotate video!")


if __name__ == "__main__":
    main()
