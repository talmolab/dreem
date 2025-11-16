"""Module containing helper functions for datasets."""

import math
from xml.etree import cElementTree as et
from typing import Dict
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
import torch
from numpy.typing import ArrayLike
from PIL import Image
from dreem.inference.boxes import Boxes
from sleap_io import LabeledFrame, Labels
from sleap_io.io.slp import (
    read_hdf5_attrs,
    read_hdf5_dataset,
    read_instances,
    read_metadata,
    read_points,
    read_pred_points,
    read_skeletons,
    read_tracks,
    read_videos,
)
from torchvision.transforms import functional as tvf


def load_slp(labels_path: str, open_videos: bool = True) -> Labels:
    """Read a SLEAP labels file.

    Args:
        labels_path: A string path to the SLEAP labels file.
        open_videos: If `True` (the default), attempt to open the video backend for
            I/O. If `False`, the backend will not be opened (useful for reading metadata
            when the video files are not available).

    Returns:
        The processed `Labels` object.
    """
    tracks = read_tracks(labels_path)
    videos = read_videos(labels_path, open_backend=open_videos)
    skeletons = read_skeletons(labels_path)
    points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    instances = read_instances(
        labels_path, skeletons, tracks, points, pred_points, format_id
    )
    metadata = read_metadata(labels_path)
    provenance = metadata.get("provenance", dict())

    frames = read_hdf5_dataset(labels_path, "frames")
    labeled_frames = []
    annotated_segments = []
    curr_segment_start = frames[0][2]
    curr_frame = curr_segment_start
    # note that frames only contains frames with labelled instances, not all frames
    for i, video_id, frame_idx, instance_id_start, instance_id_end in frames:
        # if no instances, don't add this frame to the labeled frames
        if len(instances[instance_id_start:instance_id_end]) == 0:
            continue

        labeled_frames.append(
            LabeledFrame(
                video=videos[video_id],
                frame_idx=int(frame_idx),
                instances=instances[instance_id_start:instance_id_end],
            )
        )
        if frame_idx == curr_frame:
            pass
        elif frame_idx == curr_frame + 1:
            curr_frame = frame_idx
        elif frame_idx > curr_frame + 1:
            annotated_segments.append((curr_segment_start, curr_frame))
            curr_segment_start = frame_idx
            curr_frame = frame_idx

    # add last segment
    annotated_segments.append((curr_segment_start, curr_frame))

    labels = Labels(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=skeletons,
        tracks=tracks,
        provenance=provenance,
    )
    labels.provenance["filename"] = labels_path

    return labels, annotated_segments


def is_pose_centroid_only(pose: Dict[str, torch.Tensor]) -> bool:
    """Check if a pose contains only a single key named "centroid".

    Args:
        pose: a pose as a dictionary mapping keypoint names to 2D coordinates.

    Returns:
        bool: True if the pose contains only a single key named "centroid", False otherwise.
    """
    if len(pose.keys()) <= 1 and pose.get("centroid") is not None:
        return True
    return False


def gather_pose_array(poses: list[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Collate a list of pose dictionaries into a torch tensor of shape (N, num_keys, 2).

    Each pose in the input list should be a dict mapping keypoint names to 2D coordinates (2-vector).

    Args:
        poses: List of dicts, where each dict {key: value} contains keypoint names as keys and 2D coordinates as values.

    Returns:
        torch.Tensor: Tensor of shape (N, num_keys, 2) containing the collated poses, where N is the number of poses and num_keys is the maximum number of keys across all poses.
    """
    num_pose_keys = [len(instance.keys()) for instance in poses]
    max_num_keys = max(num_pose_keys)
    pose_arr = torch.full((len(poses), max_num_keys, 2), fill_value=torch.nan)
    for i, instance in enumerate(poses):
        for j, node_name in enumerate(instance.keys()):
            pose_arr[i, j, :] = torch.tensor(instance[node_name])
    return pose_arr


def pad_bbox(bbox: ArrayLike, padding: int = 16) -> torch.Tensor:
    """Pad bounding box coordinates.

    Args:
        bbox: Bounding box in [y1, x1, y2, x2] format.
        padding: Padding to add to each side in pixels.

    Returns:
        Padded bounding box in [y1, x1, y2, x2] format.
    """
    y1, x1, y2, x2 = bbox
    y1, x1 = y1 - padding, x1 - padding
    y2, x2 = y2 + padding, x2 + padding
    return torch.Tensor([y1, x1, y2, x2])


def crop_bbox(img: torch.Tensor, bbox: ArrayLike) -> torch.Tensor:
    """Crop an image to a bounding box.

    Args:
        img: Image as a tensor of shape (channels, height, width).
        bbox: Bounding box in [y1, x1, y2, x2] format.

    Returns:
        Cropped pixels as tensor of shape (channels, height, width).
    """
    # Crop to the bounding box.
    y1, x1, y2, x2 = bbox
    crop = tvf.crop(
        img,
        top=int(y1.round()),
        left=int(x1.round()),
        height=int((y2 - y1).round()),
        width=int((x2 - x1).round()),
    )

    return crop


def get_mask_from_keypoints(
    arr_pose: np.ndarray,
    crop: torch.Tensor,
    dilation_radius_px: int,
    bbox: torch.Tensor,
) -> torch.Tensor:
    """Get a mask from keypoints.

    Args:
        arr_pose: array of keypoints
        crop: crop of the image
        dilation_radius_px: radius of the dilation in pixels
        bbox: bounding box of the crop
    Returns:
        mask: mask of the image
    """
    y1, x1, y2, x2 = bbox.numpy()
    arr_pose_transformed = arr_pose.copy()
    arr_pose_transformed[:, 0] = arr_pose_transformed[:, 0] - x1
    arr_pose_transformed[:, 1] = arr_pose_transformed[:, 1] - y1
    X, Y = np.meshgrid(np.arange(crop.shape[2]), np.arange(crop.shape[1]))
    dists = np.sqrt(
        (X[..., None] - arr_pose_transformed[:, 0]) ** 2
        + (Y[..., None] - arr_pose_transformed[:, 1]) ** 2
    )
    mask = np.min(dists, axis=-1) < dilation_radius_px
    mask = torch.from_numpy(mask.astype(np.uint8))
    return mask


def get_pose_principal_axis(pose_arr: torch.Tensor) -> torch.Tensor:
    """Get the principal axis of a pose.

    Args:
        pose_arr: a tensor of shape (N, num_keys, 2)

    Returns:
        The principal axis of the pose
    """
    # each instance can have a different number of pose keypoints
    principal_axes = []
    for instance in pose_arr:
        valid = ~torch.isnan(instance).any(dim=-1)
        instance_filt = instance[valid]
        U, S, Vt = torch.linalg.svd(instance_filt - instance_filt.mean(dim=0))
        principal_axes.append(Vt[:, 0])
    return torch.stack(principal_axes)


def pad_variable_size_crops(instance, target_size):
    """Pad or crop an instance's crop to the target size.

    Args:
        instance: Instance object with a crop attribute
        target_size: Tuple of (height, width) for the target size

    Returns:
        The instance with modified crop
    """
    _, c, h, w = instance.crop.shape
    target_h, target_w = target_size

    # Crop the image further if target_size is smaller than current crop size
    if h > target_h or w > target_w:
        instance.crop = tvf.center_crop(
            instance.crop, (min(h, target_h), min(w, target_w))
        )

    _, c, h, w = instance.crop.shape

    if h < target_h or w < target_w:
        # If height or width is smaller than target size, pad the image to target_size
        pad_w = max(0, target_w - w)
        pad_h = max(0, target_h - h)

        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top

        # Apply padding
        instance.crop = tvf.pad(
            instance.crop,
            (pad_w_left, pad_h_top, pad_w_right, pad_h_bottom),
            0,
            "constant",
        )

    return instance


def _pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute the intersection area between __all__ N x M pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1: First set of boxes (Boxes object containing N boxes).
        boxes2: Second set of boxes (Boxes object containing M boxes).

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, :, 2:], boxes2[:, :, 2:]) - torch.max(
        boxes1[:, None, :, :2], boxes2[:, :, :2]
    )  # [N,M,n_anchors,     2]
    width_height.clamp_(min=0)  # [N,M, n_anchors, 2]

    intersection = width_height.prod(dim=3)  # [N,M, n_anchors]

    return intersection


def _pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute intersection over union between all N x M pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1: First set of boxes (Boxes object containing N boxes).
        boxes2: Second set of boxes (Boxes object containing M boxes).

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]

    inter = _pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter >= 0,
        inter / (area1[:, None, :] + area2 - inter),
        torch.nan,
    )
    return iou.nanmean(dim=-1)


def pairwise_iom(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute the intersection over minimum area between all N x M pairs of boxes."""
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = _pairwise_intersection(boxes1, boxes2)
    iom = torch.where(
        inter >= 0,
        inter / torch.min(area1, area2),
        torch.nan,
    )
    return iom.nanmean(dim=-1)


def nms(ioms: torch.Tensor, threshold: float) -> list[int]:
    """Non-maximum suppression.

    Args:
        ioms: IoM matrix
        threshold: threshold for non-maximum suppression
    Returns:
        list of indices of the boxes to keep
    """
    keep_inds = []
    x, y = np.where(ioms > threshold)
    coords = np.stack([x, y], axis=-1)
    self_mask = coords[:, 0] == coords[:, 1]  # diagonal elements are self-ious
    coords = coords[~self_mask]

    return coords


def get_bbox(center: ArrayLike, size: int | tuple[int]) -> torch.Tensor:
    """Get a square bbox around a centroid coordinates.

    Args:
        center: centroid coordinates in (x,y)
        size: size of the bounding box

    Returns:
        A torch tensor in form y1, x1, y2, x2
    """
    if isinstance(size, int):
        size = (size, size)
    cx, cy = center[0], center[1]

    y1 = max(0, -size[-1] // 2 + cy)
    x1 = max(0, -size[0] // 2 + cx)
    y2 = size[-1] // 2 + cy if y1 != 0 else size[1]
    x2 = size[0] // 2 + cx if x1 != 0 else size[0]
    bbox = torch.Tensor([y1, x1, y2, x2])

    return bbox


def get_tight_bbox(pose: ArrayLike) -> torch.Tensor:
    """Get a tight bbox around an instance.

    Args:
        pose: array of keypoints around which to create the tight bbox

    Returns:
        A torch tensor in form y1, x1, y2, x2 representing the tight bbox
    """
    x_coords = pose[:, 0]
    y_coords = pose[:, 1]
    x1 = np.min(x_coords)
    x2 = np.max(x_coords)
    y1 = np.min(y_coords)
    y2 = np.max(y_coords)
    bbox = torch.Tensor([y1, x1, y2, x2])

    return bbox


def get_tight_bbox_masks(mask: ArrayLike) -> torch.Tensor:
    """Get a tight bbox around an instance.

    Args:
        mask: mask of the instance

    Returns:
        A torch tensor in form y1, x1, y2, x2 representing the tight bbox
    """
    max_x = np.asarray(mask != 0).nonzero()[1].max()
    max_y = np.asarray(mask != 0).nonzero()[0].max()
    min_x = np.asarray(mask != 0).nonzero()[1].min()
    min_y = np.asarray(mask != 0).nonzero()[0].min()
    bbox = torch.Tensor([min_y, min_x, max_y, max_x])

    return bbox


def centroid_bbox(points: ArrayLike, anchors: list, crop_size: int) -> torch.Tensor:
    """Calculate bbox around instance centroid.

    This is useful for ensuring that crops are centered around each instance
    in the case of incorrect pose estimates.

    Args:
        points: 2d array of centroid coordinates where each row corresponds to a
            different anchor point.
        anchors: indices of a given anchor point to use as the centroid
        crop_size: Integer specifying the crop height and width

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    for anchor in anchors:
        cx, cy = points[anchor][0], points[anchor][1]
        if not np.isnan(cx):
            break

    bbox = torch.Tensor(
        [
            -crop_size / 2 + cy,
            -crop_size / 2 + cx,
            crop_size / 2 + cy,
            crop_size / 2 + cx,
        ]
    )

    return bbox


def pose_bbox(points: np.ndarray, bbox_size: tuple[int] | int) -> torch.Tensor:
    """Calculate bbox around instance pose.

    Args:
        points: an np array of shape nodes x 2,
        bbox_size: size of bbox either an int indicating square bbox or in (x,y)

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    if isinstance(bbox_size, int):
        bbox_size = (bbox_size, bbox_size)

    c = np.nanmean(points, axis=0)
    bbox = torch.Tensor(
        [
            c[-1] - bbox_size[-1] / 2,
            c[0] - bbox_size[0] / 2,
            c[-1] + bbox_size[-1] / 2,
            c[0] + bbox_size[0] / 2,
        ]
    )
    return bbox


def resize_and_pad(img: torch.Tensor, output_size: int) -> torch.Tensor:
    """Resize and pad an image to fit a square output size.

    Args:
        img: Image as a tensor of shape (channels, height, width).
        output_size: Integer size of height and width of output.

    Returns:
        The image zero padded to be of shape (channels, output_size, output_size).
    """
    # Figure out how to scale without breaking aspect ratio.
    img_height, img_width = img.shape[-2:]
    if img_width < img_height:  # taller
        crop_height = output_size
        scale = crop_height / img_height
        crop_width = int(img_width * scale)
    else:  # wider
        crop_width = output_size
        scale = crop_width / img_width
        crop_height = int(img_height * scale)

    # Scale without breaking aspect ratio.
    img = tvf.resize(img, size=[crop_height, crop_width])

    # Pad to square.
    img_height, img_width = img.shape[-2:]
    hp1 = int((output_size - img_width) / 2)
    vp1 = int((output_size - img_height) / 2)
    hp2 = output_size - (img_width + hp1)
    vp2 = output_size - (img_height + vp1)
    padding = (hp1, vp1, hp2, vp2)
    return tvf.pad(img, padding, 0, "constant")


class NodeDropout:
    """Node dropout augmentation.

    Drop up to `n` nodes with probability `p`.
    """

    def __init__(self, p: float, n: int) -> None:
        """Initialize Node Dropout Augmentation.

        Args:
            p: the probability with which to drop the nodes
            n: the maximum number of nodes to drop
        """
        self.n = n
        self.p = p

    def __call__(self, nodes: list[str]) -> list[str]:
        """Wrap `drop_nodes` to enable class call.

        Args:
            nodes: A list of available node names to drop.

        Returns:
            dropped_nodes: A list of up to `self.n` nodes to drop.
        """
        return self.forward(nodes)

    def forward(self, nodes: list[str]) -> list[str]:
        """Drop up to `n` random nodes with probability p.

        Args:
            nodes: A list of available node names to drop.

        Returns:
            dropped_nodes: A list of up to `self.n` nodes to drop.
        """
        if self.n == 0 or self.p == 0:
            return []

        nodes_to_drop = np.random.permutation(nodes)
        node_dropout_p = np.random.uniform(size=len(nodes_to_drop))

        dropped_node_inds = np.where(node_dropout_p < self.p)
        node_dropout_p = node_dropout_p[dropped_node_inds]

        n_nodes_to_drop = min(self.n, len(node_dropout_p))

        dropped_node_inds = np.argpartition(node_dropout_p, -n_nodes_to_drop)[
            -n_nodes_to_drop:
        ]

        dropped_nodes = nodes_to_drop[dropped_node_inds]

        return dropped_nodes


def sorted_anchors(labels: sio.Labels) -> list[str]:
    """Sort anchor names from most instances with that node to least.

    Args:
        labels: a sleap_io.labels object containing all the labels for that video

    Returns:
        A list of anchor names sorted by most nodes to least nodes
    """
    all_anchors = labels.skeletons[0].node_names

    anchor_counts = {anchor: 0 for anchor in all_anchors}

    for i in range(len(labels)):
        lf = labels[i]
        for instance in lf:
            for anchor in all_anchors:
                x, y = instance[anchor].x, instance[anchor].y
                if np.isnan(x) or np.isnan(y):
                    anchor_counts[anchor] += 1

    sorted_anchors = sorted(anchor_counts.keys(), key=lambda k: anchor_counts[k])

    return sorted_anchors


def parse_trackmate(data_path: str) -> pd.DataFrame:
    """Parse trackmate xml or csv labels file.

    Logic adapted from https://github.com/hadim/pytrackmate.

    Args:
        data_path: string path to xml or csv file storing trackmate trajectory labels

    Returns:
        `pandas DataFrame` containing frame number, track_ids,
        and centroid x,y coordinates in pixels
    """
    if data_path.endswith(".xml"):
        root = et.fromstring(open(data_path).read())

        objects = []
        features = root.find("Model").find("FeatureDeclarations").find("SpotFeatures")
        features = [c.get("feature") for c in list(features)] + ["ID"]

        spots = root.find("Model").find("AllSpots")

        objects = []

        for frame in spots.findall("SpotsInFrame"):
            for spot in frame.findall("Spot"):
                single_object = []
                for label in features:
                    single_object.append(spot.get(label))
                objects.append(single_object)

        tracks_df = pd.DataFrame(objects, columns=features)
        tracks_df = tracks_df.astype(np.float)

        filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in root.find("Model").find("FilteredTracks").findall("TrackID")
        ]

        label_id = 0
        tracks_df["label"] = np.nan

        tracks = root.find("Model").find("AllTracks")
        for track in tracks.findall("Track"):
            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:
                spot_ids = [
                    (
                        edge.get("SPOT_SOURCE_ID"),
                        edge.get("SPOT_TARGET_ID"),
                        edge.get("EDGE_TIME"),
                    )
                    for edge in track.findall("Edge")
                ]
                spot_ids = np.array(spot_ids).astype("float")[:, :2]
                spot_ids = set(spot_ids.flatten())

                tracks_df.loc[tracks_df["ID"].isin(spot_ids), "TRACK_ID"] = label_id
                label_id += 1

    elif data_path.endswith(".csv"):
        tracks_df = pd.read_csv(data_path, encoding="ISO-8859-1")

    else:
        raise ValueError(f"Unsupported trackmate file extension: {data_path}")

    tracks_df = tracks_df.apply(pd.to_numeric, errors="coerce", downcast="integer")

    posx_key = "POSITION_X"
    posy_key = "POSITION_Y"
    frame_key = "FRAME"
    track_key = "TRACK_ID"

    mapper = {
        "X": posx_key,
        "Y": posy_key,
        "x": posx_key,
        "y": posy_key,
        "Slice n°": frame_key,
        "Track n°": track_key,
    }

    if "t" in tracks_df:
        mapper.update({"t": frame_key})

    tracks_df = tracks_df.rename(mapper=mapper, axis=1)

    if data_path.endswith(".csv"):
        # 0 index track and frame ids
        if min(tracks_df[frame_key]) == 1:
            tracks_df[frame_key] = tracks_df[frame_key] - 1

        if min(tracks_df[track_key] == 1):
            tracks_df[track_key] = tracks_df[track_key] - 1

    return tracks_df


def parse_synthetic(xml_path: str, source: str = "icy") -> pd.DataFrame:
    """Parse .xml labels from synthetic data generated by ICY or ISBI tracking challenge.

    Logic adapted from https://github.com/sylvainprigent/napari-tracks-reader/blob/main/napari_tracks_reader

    Args:
        xml_path: path to .xml file containing ICY or ISBI gt trajectory labels
        source: synthetic dataset type. Should be either icy or isbi

    Returns:
        pandas DataFrame containing frame idx, gt track id
        and centroid x,y coordinates in pixels
    """
    if source.lower() == "icy":
        root_tag = "trackgroup"
    elif source.lower() == "isbi":
        root_tag = "TrackContestISBI2012"
    else:
        raise ValueError(f"{source} source mode not supported")

    tree = et.parse(xml_path)

    root = tree.getroot()
    tracks = np.empty((0, 4))

    # get the trackgroup element
    idx_trackgroup = 0
    for i in range(len(root)):
        if root[i].tag == root_tag:
            idx_trackgroup = i
            break

    ids_map = {}
    track_id = -1
    for track_element in root[idx_trackgroup]:
        track_id += 1

        try:
            ids_map[track_element.attrib["id"]] = track_id
        except KeyError:
            pass

        for detection_element in track_element:
            row = [
                float(track_id),
                float(detection_element.attrib["t"]),
                float(detection_element.attrib["y"]),
                float(detection_element.attrib["x"]),
            ]
            tracks = np.concatenate((tracks, [row]), axis=0)

    tracks_df = pd.DataFrame(
        tracks, columns=["TRACK_ID", "FRAME", "POSITION_Y", "POSITION_X"]
    )

    tracks_df = tracks_df.apply(pd.to_numeric, errors="coerce", downcast="integer")

    return tracks_df


class LazyTiffStack:
    """Class used for loading tiffs without loading into memory."""

    def __init__(self, filename: str):
        """Initialize class.

        Args:
            filename: name of tif file to be opened
        """
        # expects spatial, channels
        self.image = Image.open(filename)

    def __getitem__(self, section_idx: int) -> Image:
        """Get frame.

        Args:
            section_idx: index of frame or z-slice to get.

        Returns:
            a PIL image of that frame/z-slice.
        """
        self.image.seek(section_idx)
        return self.image

    def get_section(self, section_idx: int) -> np.array:
        """Get frame as ndarray.

        Args:
            section_idx: index of frame or z-slice to get.

        Returns:
            an np.array of that frame/z-slice.
        """
        section = self.__getitem__(section_idx)
        return np.array(section)

    def close(self):
        """Close tiff stack."""
        self.file.close()


def build_augmentations(augmentations: dict) -> A.Compose:
    """Get augmentations for dataset.

    Args:
        augmentations: a dict containing the name of the augmentations
                       and their parameters

    Returns:
        An Albumentations composition of different augmentations.
    """
    aug_list = []
    for aug_name, aug_args in augmentations.items():
        aug_class = getattr(A, aug_name)
        aug = aug_class(**aug_args)
        aug_list.append(aug)

    augs = A.Compose(
        aug_list,
        p=1.0,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    return augs


def get_max_padding(height: int, width: int) -> tuple:
    """Calculate maximum padding dimensions for a given height and width.

    Useful if padding is required for rotational augmentations, e.g when
    centroids lie on the borders of an image.

    Args:
        height: The original height.
        width: The original width.

    Returns:
        A tuple containing the padded height and padded width.
    """
    diagonal = math.ceil(math.sqrt(height**2 + width**2))

    padded_height = height + (diagonal - height)
    padded_width = width + (diagonal - width)

    return padded_height, padded_width


def view_training_batch(
    instances: list[dict[str, list[np.ndarray]]], num_frames: int = 1, cmap=None
) -> None:
    """Display a grid of images from a batch of training instances.

    Args:
        instances: A list of training instances, where each instance is a
            dictionary containing the object crops.
        num_frames: The number of frames to display per instance.
        cmap: Optional colormap to use for displaying images.

    Returns:
        None
    """
    num_crops = len(instances[0]["crops"])
    num_columns = num_crops
    num_rows = num_frames

    base_size = 2
    fig_size = (base_size * num_columns, base_size * num_rows)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=fig_size)

    for i in range(num_frames):
        for j, data in enumerate(instances[i]["crops"]):
            try:
                ax = (
                    axes[j]
                    if num_frames == 1
                    else (axes[i] if num_crops == 1 else axes[i, j])
                )

                (ax.imshow(data.T) if cmap is None else ax.imshow(data.T, cmap=cmap))
                ax.axis("off")

            except Exception as e:
                print(e)
                pass

    plt.tight_layout()
    plt.show()
