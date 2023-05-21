"""Module containing helper functions for datasets."""
from PIL import Image
from numpy.typing import ArrayLike
from torchvision.transforms import functional as tvf
from xml.etree import cElementTree as et
import albumentations as A
import numpy as np
import pandas as pd
import sleap_io as sio
import torch


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
        bbox: Bounding box in [x1, y1, x2, y2] format.

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


def get_bbox(center: ArrayLike, size: int) -> torch.Tensor:
    """Get a square bbox around a centroid coordinates.

    Args:
        center: centroid coordinates in (x,y)
        size: size of the bounding box

    Returns:
        A torch tensor in form y1, x1, y2, x2
    """
    cx, cy = center[0], center[1]

    bbox = torch.Tensor(
        [-size // 2 + cy, -size // 2 + cx, size // 2 + cy, size // 2 + cx]
    )

    return bbox


def centroid_bbox(
    instance: sio.Instance, anchors: list, crop_size: int
) -> torch.Tensor:
    """Calculate bbox around instance centroid.

    This is useful for ensuring that crops are centered around each instance
    in the case of incorrect pose estimates.

    Args:
        instance: a labeled instance in a frame
        anchors: indices of a given anchor point to use as the centroid
        crop_size: Integer specifying the crop height and width

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    for anchor in anchors:
        cx, cy = instance[anchor][0], instance[anchor][1]
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


def pose_bbox(
    instance: sio.Instance, padding: int, im_shape: ArrayLike
) -> torch.Tensor:
    """Calculate bbox around instance pose.

    Args:
        instance: a labeled instance in a frame,
        padding: the amount to pad around the pose crop
        im_shape: the size of the original image in (w,h)

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    w, h = im_shape

    points = torch.Tensor([[p.x, p.y] for p in instance.points])

    min_x = max(torch.nanmin(points[:, 0]) - padding, 0)
    min_y = max(torch.nanmin(points[:, 1]) - padding, 0)
    max_x = min(torch.nanmax(points[:, 0]) + padding, w)
    max_y = min(torch.nanmax(points[:, 1]) + padding, h)

    bbox = torch.Tensor([min_y, min_x, max_y, max_x])
    return bbox


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


"""
PARSERS
"""


def parse_trackmate_xml(xml_path: str) -> pd.DataFrame:
    """Parse trackmate XML labels file.

    Logic adapted from https://github.com/hadim/pytrackmate.

    Args:
        xml_path: string path to xml file storing trackmate trajectory labels

    Returns:
        `pandas DataFrame` containing frame number, track_ids,
        and centroid x,y coordinates in pixels
    """
    root = et.fromstring(open(xml_path).read())

    objects = []
    features = root.find("Model").find("FeatureDeclarations").find("SpotFeatures")
    features = [c.get("feature") for c in list(features)] + ["ID"]

    spots = root.find("Model").find("AllSpots")
    trajs = pd.DataFrame([])
    objects = []
    for frame in spots.findall("SpotsInFrame"):
        for spot in frame.findall("Spot"):
            single_object = []
            for label in features:
                single_object.append(spot.get(label))
            objects.append(single_object)

    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(np.float)

    filtered_track_ids = [
        int(track.get("TRACK_ID"))
        for track in root.find("Model").find("FilteredTracks").findall("TrackID")
    ]

    label_id = 0
    trajs["label"] = np.nan

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

            trajs.loc[trajs["ID"].isin(spot_ids), "TRACK_ID"] = label_id
            label_id += 1
    trajs = trajs.apply(pd.to_numeric, errors="coerce", downcast="integer")
    posx_key = "POSITION_X"
    posy_key = "POSITION_Y"
    frame_key = "FRAME"
    track_key = "TRACK_ID"
    trajs = trajs.rename(
        mapper={
            "X": posx_key,
            "Y": posy_key,
            "x": posx_key,
            "y": posy_key,
            "Slice n°": frame_key,
            "Track n°": track_key,
        },
        axis=1,
    )
    return trajs


def parse_trackmate_csv(csv_path: str) -> pd.DataFrame:
    """Parse trackmate .csv trajectory labels file.

    Args:
        csv_path: path to trackmate .csv trajectory labels file

    Returns:
        `pandas DataFrame containing frame index, gt track id
        and centroid x,y coordinates in pixels
    """
    track = pd.read_csv(csv_path, encoding="ISO-8859-1")
    track = track.apply(pd.to_numeric, errors="coerce", downcast="integer")
    posx_key = "POSITION_X"
    posy_key = "POSITION_Y"
    frame_key = "FRAME"
    track_key = "TRACK_ID"
    track = track.rename(
        mapper={
            "X": posx_key,
            "Y": posy_key,
            "x": posx_key,
            "y": posy_key,
            "Slice n°": frame_key,
            "Track n°": track_key,
            "t": frame_key,
        },
        axis=1,
    )
    # 0 index track and frame ids
    if min(track[frame_key]) == 1:
        track[frame_key] = track[frame_key] - 1
    if min(track[track_key] == 1):
        track[track_key] = track[track_key] - 1
    return track


def parse_trackmate(trackmate_labels: str) -> pd.DataFrame:
    """Wrapper around `parse_trackmate_csv` and `parse_trackmate_xml`."""
    if ".xml" in trackmate_labels:
        return parse_trackmate_xml(trackmate_labels)
    elif ".csv" in trackmate_labels:
        return parse_trackmate_csv(trackmate_labels)
    else:
        raise ValueError("Trackmate labels file must be a `.xml` or `.csv`!")


def parse_ICY(xml_path: str) -> pd.DataFrame:
    """Parse .xml labels file from synthetic data generated by ICY.

    Logic adapted from https://github.com/sylvainprigent/napari-tracks-reader/blob/main/napari_tracks_reader/_icy_io.py.

    Args:
        xml_path: path to .xml file containing ICY gt trajectory labels

    Returns:
        pandas DataFrame containing frame idx, gt track id
        and centroid x,y coordinates in pixels
    """
    tree = et.parse(xml_path)

    root = tree.getroot()
    tracks = np.empty((0, 4))

    # get the trackgroup element
    idx_trackgroup = 0
    for i in range(len(root)):
        if root[i].tag == "trackgroup":
            idx_trackgroup = i
            break

    ids_map = {}
    track_id = -1
    for track_element in root[idx_trackgroup]:
        track_id += 1
        ids_map[track_element.attrib["id"]] = track_id
        for detection_element in track_element:
            row = [
                float(track_id),
                float(detection_element.attrib["t"]),
                float(detection_element.attrib["y"]),
                float(detection_element.attrib["x"]),
            ]
            tracks = np.concatenate((tracks, [row]), axis=0)

    trajs = pd.DataFrame(
        tracks, columns=["TRACK_ID", "FRAME", "POSITION_Y", "POSITION_X"]
    )
    trajs = trajs.apply(pd.to_numeric, errors="coerce", downcast="integer")
    return trajs


# todo: consolidate ICY / ISBY parser into one (only differ by a couple lines)
def parse_ISBI(xml_file: str) -> pd.DataFrame:
    """Parse .xml labels file from ISBI particle tracing challenge.

    logic adapted from https://github.com/sylvainprigent/napari-tracks-reader/blob/main/napari_tracks_reader/_isbi_io.py.

    Args:
        xml_file: path to .xml labels file containing gt trajectory ids from ISBI

    Returns:
        pandas DataFrame containing frame idx, gt track id and
        centroid x,y coordinates in pixels
    """
    tree = et.parse(xml_file)
    root = tree.getroot()

    tracks = np.empty((0, 4))

    # get the trackgroup element
    idx_trackcontest = 0
    for i in range(len(root)):
        if root[i].tag == "TrackContestISBI2012":
            idx_trackcontest = i
            break

    # parse tracks=particles
    track_id = -1
    for particle_element in root[idx_trackcontest]:
        track_id += 1
        for detection_element in particle_element:
            row = [
                float(track_id),
                float(detection_element.attrib["t"]),
                float(detection_element.attrib["y"]),
                float(detection_element.attrib["x"]),
            ]
            tracks = np.concatenate((tracks, [row]), axis=0)

    trajs = pd.DataFrame(
        tracks, columns=["TRACK_ID", "FRAME", "POSITION_Y", "POSITION_X"]
    )
    trajs = trajs.apply(pd.to_numeric, errors="coerce", downcast="integer")
    return trajs


# todo: type
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


# todo: type
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
    return A.Compose(aug_list, p=1.0, keypoint_params=A.KeypointParams(format="xy"))
