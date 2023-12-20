"""Module containing helper functions for datasets."""
from PIL import Image
from numpy.typing import ArrayLike
from torchvision.transforms import functional as tvf
from typing import List, Dict, Union
from xml.etree import cElementTree as et
import albumentations as A
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
import torch
import time


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


def crop_bbox(
    img: ArrayLike, bbox: ArrayLike, pose: ArrayLike = None, mask=None, size: Union[int, tuple[int]] = None
) -> tuple[np.ndarray]:
    """Crop an image to a bounding box.

    Args:
        img: Image as ArrayLike Object of shape (channels, height, width).
        bbox: Bounding box in [y1, x1, y2, x2] format.
        pose: Pose in shape (n, 2) where n is the number of keypoints in the skeleton
        mask: mask of instances in shape (height, width)
        size: optional. Size of the bounding box. If size is not none then use left, top + size instead of bbox coords

    Returns:
        Cropped pixels as tensor of shape (channels, height, width) along with keypoints and masks adjusted accordingly.
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if isinstance(pose, torch.Tensor):
        pose = pose.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    if mask is None:
        aug_mask = np.zeros_like(img)
    else:
        aug_mask = mask
    if pose is None:
        aug_pose = np.zeros((1, 2))
    else:
        aug_pose = pose
    # Crop to the bounding box.
    y1, x1, y2, x2 = bbox
   
    if size is not None:
        if not isinstance(size, tuple):
            size = (size, size)
            
        y1 = int(y1)
        x1 = int(x1)
        
        y2 = y1 + size[1]
        x2 = x1 + size[0]
    crop = A.Compose(
        [
            A.augmentations.crops.transforms.Crop(
                x_min=int(np.floor(x1)), y_min=int(np.floor(y1)), x_max=int(np.floor(x2)), y_max=int(np.floor(y2))
            )
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    cropped = crop(image=img, keypoints=aug_pose, mask=aug_mask)

    aug_img, aug_pose, aug_mask = (
        cropped["image"],
        cropped["keypoints"],
        cropped["mask"],
    )

    return aug_img, aug_pose, aug_mask

def check_bbox(bbox: ArrayLike, img_shape: ArrayLike, size: tuple[int]):
    """Check that bbox edges do not exceed image shape.
    
    If y1 or x1 are negative then they become 0 and x2, y2 = img_shape. 
    If y2 or x2 > img_shape then they become img shape and y1, x2 = img_shape-size
    
    Args:
        bbox: (1,4) array containing bbox coordinates
        img_shape: The shape of the original image which the bbox is being drawn for in (C, H, W)
        size: the size of the bbox in (x,y)
    Returns:
        The adjusted bbox.
    """
    if len(img_shape) < 3:
        h, w = img_shape
    elif img_shape[-1] <=3:
        h, w, _ = img_shape
    else:
        _, h, w = img_shape
    
    y1, x1, y2, x2 = bbox
    
    if y1 < 0:
        y1 = 0
        y2 = size[-1]
    if x1 < 0:
        x1 = 0
        x2 = size[0]
    
    if y2 > h:
        y2 = h
        y1 = y2 - size[-1]
    
    if x2 > w:
        x2 = w
        x1 = x2 - size[0]
    return [y1, x1, y2, x2]
def get_bbox(center: ArrayLike, size: Union[int, tuple[int]],img_shape: ArrayLike, padding: int = 0) -> torch.Tensor:
    """Get a square bbox around a centroid coordinates.

    Args:
        center: centroid coordinates in (x,y)
        size: size of the bounding box
        padding: amount of padding around bbox

    Returns:
        A torch tensor in form y1, x1, y2, x2
    """
    if type(size) == int:
        size = (size + 2*padding, size + 2*padding)
    else:
        size = (s_i + 2*padding for s_i in size)
        
    cx, cy = center[0], center[1]
    
    y1 = -size[-1] // 2 + cy
    x1 = -size[0] // 2 + cx
    y2 = size[-1] // 2 + cy
    x2= size[0] // 2 + cx
    
    
    bbox = torch.Tensor(
        [y1, x1, y2, x2]
    )

    return check_bbox(bbox, img_shape, size)


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


def pose_bbox(points: np.ndarray, bbox_size: Union[tuple[int], int], img_shape, padding = 0) -> torch.Tensor:
    """Calculate bbox around instance pose.

    Args:
        instance: a labeled instance in a frame,
        bbox_size: size of bbox either an int indicating square bbox or in (x,y)

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    if type(bbox_size) == int:
        bbox_size = (bbox_size + 2*padding, bbox_size + 2*padding)
    else:
        bbox_size = (s_i + padding for s_i in bbox_size)
    # print(points)
    minx = np.nanmin(points[:, 0], axis=-1)
    miny = np.nanmin(points[:, -1], axis=-1)
    minpoints = np.array([minx, miny]).T

    maxx = np.nanmax(points[:, 0], axis=-1)
    maxy = np.nanmax(points[:, -1], axis=-1)
    maxpoints = np.array([maxx, maxy]).T

    c = (minpoints + maxpoints) / 2

    bbox = torch.Tensor(
        [
            c[-1] - bbox_size[-1] / 2,
            c[0] - bbox_size[0] / 2,
            c[-1] + bbox_size[-1] / 2,
            c[0] + bbox_size[0] / 2,
        ]
    )
    return check_bbox(bbox, img_shape, bbox_size)


def resize_and_pad(img: torch.Tensor, output_size: int):
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


def expand_to_rank(
    x: torch.Tensor, target_rank: int, prepend: bool = True
) -> torch.Tensor:
    """Expand a tensor to a target rank by adding singleton dimensions.

    Args:
        x: Any `torch.Tensor` with rank <= `target_rank`. If the rank is higher than
            `target_rank`, the tensor will be returned with the same shape.
        target_rank: Rank to expand the input to.
        prepend: If True, singleton dimensions are added before the first axis of the
            data. If False, singleton dimensions are added after the last axis.

    Returns:
        The expanded tensor of the same dtype as the input, but with rank `target_rank`.

        The output has the same exact data as the input tensor and will be identical if
        they are both flattened.
    """
    if len(x.shape) < 2:
        source_rank = torch.linalg.matrix_rank(x.unsqueeze(0))
    else:
        source_rank = torch.linalg.matrix_rank(x)

    source_rank = source_rank.max()
    # print(target_rank, source_rank)
    n_singleton_dims = torch.maximum(target_rank - source_rank, torch.tensor(0)).item()

    singleton_dims = torch.ones((n_singleton_dims), dtype=torch.int32)
    if prepend:
        new_shape = torch.concat([singleton_dims, torch.tensor(x.shape)], axis=0)
    else:
        new_shape = torch.concat([torch.tensor(x.shape), singleton_dims], axis=0)
    return x.view(*new_shape)


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
        except:
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


def build_augmentations(augmentations: dict):
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
        # is_check_shapes=False,
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
    instances: List[Dict[str, List[np.ndarray]]], num_frames: int = 1, cmap=None
) -> None:
    """Display a grid of images from a batch of training instances.

    Args:
        instances: A list of training instances, where each instance is a
            dictionary containing the object crops.
        num_frames: The number of frames to display per instance.

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

                ax.imshow(data.T) if isinstance(cmap, None) else ax.imshow(
                    data.T, cmap=cmap
                )
                ax.axis("off")

            except Exception as e:
                print(e)
                pass

    plt.tight_layout()
    plt.show()


class Timer:
    """Timer class for profiling operations."""

    def __init__(self, verbose=False):
        """Initialize timer.

        Args:
            verbose: Whether or not to print the time elapsed for each `time` call.
        """
        self.curr_time = time.perf_counter()
        self.prev_time = self.curr_time
        self.total_time = 0
        self.verbose = verbose

    def time(self, op: str) -> float:
        """Time operation.

        Args:
            op: A string representing the operation being timed

        Returns:
            The time elapsed for operation in seconds.
        """
        self.prev_time = self.curr_time
        self.curr_time = time.perf_counter()

        time_elapsed = self.curr_time - self.prev_time
        if self.verbose:
            print(f"{op} took {time_elapsed} seconds.")
        self.total_time += time_elapsed
        return time_elapsed
