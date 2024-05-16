"""Module containing data classes such as Instances and Frames."""

import torch
import sleap_io as sio
import numpy as np
import attrs
from numpy.typing import ArrayLike
from typing import Union, List


def _to_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif np.isscalar(data):
        return torch.tensor([data])
    else:
        return torch.tensor(data)


@attrs.define(eq=False)
class Frame:
    """Data structure containing metadata for a single frame of a video.

    Attributes:
        video_id: The video index in the dataset.
        frame_id: The index of the frame in a video.
        vid_file: The path to the video the frame is from.
        img_shape: The shape of the original frame (not the crop).
        instances: A list of Instance objects that appear in the frame.
        asso_output: The association matrix between instances
        output directly from the transformer.
        matches: matches from LSA algorithm between the instances and
        available trajectories during tracking.
        traj_score: Either a dict containing the association matrix
        between instances and trajectories along postprocessing pipeline
        or a single association matrix.
        device: The device the frame should be moved to.
    """

    _video_id: int = attrs.field(alias="video_id", converter=_to_tensor)
    _frame_id: int = attrs.field(alias="frame_id", converter=_to_tensor)
    _video: str = attrs.field(alias="vid_file", default="")
    _img_shape: ArrayLike = attrs.field(
        alias="img_shape", converter=_to_tensor, factory=list
    )

    _instances: list["Instance"] = attrs.field(alias="instances", factory=list)
    _asso_output: ArrayLike = attrs.field(alias="asso_output", default=None)
    _matches: tuple = attrs.field(alias="matches", factory=tuple)
    _traj_score: dict = attrs.field(alias="traj_score", factory=dict)
    _device: str = attrs.field(alias="device", default=None)

    def __attrs_post_init__(self):
        self.to(self.device)

    def __repr__(self) -> str:
        """Return String representation of the Frame.

        Returns:
            The string representation of the frame.
        """
        return (
            "Frame("
            f"video={self._video.filename if isinstance(self._video, sio.Video) else self._video}, "
            f"video_id={self._video_id.item()}, "
            f"frame_id={self._frame_id.item()}, "
            f"img_shape={self._img_shape}, "
            f"num_detected={self.num_detected}, "
            f"asso_output={self._asso_output}, "
            f"traj_score={self._traj_score}, "
            f"matches={self._matches}, "
            f"instances={self._instances}, "
            f"device={self._device}"
            ")"
        )

    def to(self, map_location: str):
        """Move frame to different device or dtype (See `torch.to` for more info).

        Args:
            map_location: A string representing the device to move to.

        Returns:
            The frame moved to a different device/dtype.
        """
        self._video_id = self._video_id.to(map_location)
        self._frame_id = self._frame_id.to(map_location)
        self._img_shape = self._img_shape.to(map_location)

        if isinstance(self._asso_output, torch.Tensor):
            self._asso_output = self._asso_output.to(map_location)

        if isinstance(self._matches, torch.Tensor):
            self._matches = self._matches.to(map_location)

        for key, val in self._traj_score.items():
            if isinstance(val, torch.Tensor):
                self._traj_score[key] = val.to(map_location)

        for instance in self._instances:
            instance = instance.to(map_location)

        self._device = map_location
        return self

    def to_slp(
        self, track_lookup: dict[int, sio.Track] = {}
    ) -> tuple[sio.LabeledFrame, dict[int, sio.Track]]:
        """Convert Frame to sleap_io.LabeledFrame object.

        Args:
            track_lookup: A lookup dictionary containing the track_id and sio.Track for persistence

        Returns: A tuple containing a LabeledFrame object with necessary metadata and
        a lookup dictionary containing the track_id and sio.Track for persistence
        """
        slp_instances = []
        for instance in self.instances:
            slp_instance, track_lookup = instance.to_slp(track_lookup=track_lookup)
            slp_instances.append(slp_instance)
        return (
            sio.LabeledFrame(
                video=self.video,
                frame_idx=self.frame_id.item(),
                instances=slp_instances,
            ),
            track_lookup,
        )

    @property
    def device(self) -> str:
        """The device the frame is on.

        Returns:
            The string representation of the device the frame is on.
        """
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """Set the device.

        Note: Do not set `frame.device = device` normally. Use `frame.to(device)` instead.

        Args:
            device: the device the function should be on.
        """
        self._device = device

    @property
    def video_id(self) -> torch.Tensor:
        """The index of the video the frame comes from.

        Returns:
            A tensor containing the video index.
        """
        return self._video_id

    @video_id.setter
    def video_id(self, video_id: int) -> None:
        """Set the video index.

        Note: Generally the video_id should be immutable after initialization.

        Args:
            video_id: an int representing the index of the video that the frame came from.
        """
        self._video_id = torch.tensor([video_id])

    @property
    def frame_id(self) -> torch.Tensor:
        """The index of the frame in a full video.

        Returns:
            A torch tensor containing the index of the frame in the video.
        """
        return self._frame_id

    @frame_id.setter
    def frame_id(self, frame_id: int) -> None:
        """Set the frame index of the frame.

        Note: The frame_id should generally be immutable after initialization.

        Args:
            frame_id: The int index of the frame in the full video.
        """
        self._frame_id = torch.tensor([frame_id])

    @property
    def video(self) -> Union[sio.Video, str]:
        """Get the video associated with the frame.

        Returns: An sio.Video object representing the video or a placeholder string
        if it is not possible to create the sio.Video
        """
        return self._video

    @video.setter
    def video(self, video_filename: str) -> None:
        """Set the video associated with the frame.

        Note: we try to store the video in an sio.Video object.
        However, if this is not possible (e.g. incompatible format or missing filepath)
        then we simply store the string.

        Args:
            video_filename: string path to video_file
        """
        try:
            self._video = sio.Video(video_filename)
        except ValueError:
            self._video = video_filename

    @property
    def img_shape(self) -> torch.Tensor:
        """The shape of the pre-cropped frame.

        Returns:
            A torch tensor containing the shape of the frame. Should generally be (c, h, w)
        """
        return self._img_shape

    @img_shape.setter
    def img_shape(self, img_shape: ArrayLike) -> None:
        """Set the shape of the frame image.

        Note: the img_shape should generally be immutable after initialization.

        Args:
            img_shape: an ArrayLike object containing the shape of the frame image.
        """
        self._img_shape = _to_tensor(img_shape)

    @property
    def instances(self) -> List["Instance"]:
        """A list of instances in the frame.

        Returns:
            The list of instances that appear in the frame.
        """
        return self._instances

    @instances.setter
    def instances(self, instances: List["Instance"]) -> None:
        """Set the frame's instance.

        Args:
            instances: A list of Instances that appear in the frame.
        """
        self._instances = instances

    def has_instances(self) -> bool:
        """Determine whether there are instances in the frame.

        Returns:
            True if there are instances in the frame, otherwise False.
        """
        if self.num_detected == 0:
            return False
        return True

    @property
    def num_detected(self) -> int:
        """The number of instances in the frame.

        Returns:
            the number of instances in the frame.
        """
        return len(self.instances)

    @property
    def asso_output(self) -> ArrayLike:
        """The association matrix between instances outputed directly by transformer.

        Returns:
            An arraylike (n_query, n_nonquery) association matrix between instances.
        """
        return self._asso_output

    def has_asso_output(self) -> bool:
        """Determine whether the frame has an association matrix computed.

        Returns:
            True if the frame has an association matrix otherwise, False.
        """
        if self._asso_output is None or len(self._asso_output) == 0:
            return False
        return True

    @asso_output.setter
    def asso_output(self, asso_output: ArrayLike) -> None:
        """Set the association matrix of a frame.

        Args:
            asso_output: An arraylike (n_query, n_nonquery) association matrix between instances.
        """
        self._asso_output = asso_output

    @property
    def matches(self) -> tuple:
        """Matches between frame instances and availabel trajectories.

        Returns:
            A tuple containing the instance idx and trajectory idx for the matched instance.
        """
        return self._matches

    @matches.setter
    def matches(self, matches: tuple) -> None:
        """Set the frame matches.

        Args:
            matches: A tuple containing the instance idx and trajectory idx for the matched instance.
        """
        self._matches = matches

    def has_matches(self) -> bool:
        """Check whether or not matches have been computed for frame.

        Returns:
            True if frame contains matches otherwise False.
        """
        if self._matches is not None and len(self._matches) > 0:
            return True
        return False

    def get_traj_score(self, key=None) -> Union[dict, ArrayLike, None]:
        """Get dictionary containing association matrix between instances and trajectories along postprocessing pipeline.

        Args:
            key: The key of the trajectory score to be accessed.
            Can be one of {None, 'initial', 'decay_time', 'max_center_dist', 'iou', 'final'}

        Returns:
            - dictionary containing all trajectory scores if key is None
            - trajectory score associated with key
            - None if the key is not found
        """
        if key is None:
            return self._traj_score
        else:
            try:
                return self._traj_score[key]
            except KeyError as e:
                print(f"Could not access {key} traj_score due to {e}")
                return None

    def add_traj_score(self, key, traj_score: ArrayLike) -> None:
        """Add trajectory score to dictionary.

        Args:
            key: key associated with traj score to be used in dictionary
            traj_score: association matrix between instances and trajectories
        """
        self._traj_score[key] = traj_score

    def has_traj_score(self) -> bool:
        """Check if any trajectory association matrix has been saved.

        Returns:
            True there is at least one association matrix otherwise, false.
        """
        if len(self._traj_score) == 0:
            return False
        return True

    def has_gt_track_ids(self) -> bool:
        """Check if any of frames instances has a gt track id.

        Returns:
            True if at least 1 instance has a gt track id otherwise False.
        """
        if self.has_instances():
            return any([instance.has_gt_track_id() for instance in self.instances])
        return False

    def get_gt_track_ids(self) -> torch.Tensor:
        """Get the gt track ids of all instances in the frame.

        Returns:
            an (N,) shaped tensor with the gt track ids of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.gt_track_id for instance in self.instances])

    def has_pred_track_ids(self) -> bool:
        """Check if any of frames instances has a pred track id.

        Returns:
            True if at least 1 instance has a pred track id otherwise False.
        """
        if self.has_instances():
            return any([instance.has_pred_track_id() for instance in self.instances])
        return False

    def get_pred_track_ids(self) -> torch.Tensor:
        """Get the pred track ids of all instances in the frame.

        Returns:
            an (N,) shaped tensor with the pred track ids of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.pred_track_id for instance in self.instances])

    def has_bboxes(self) -> bool:
        """Check if any of frames instances has a bounding box.

        Returns:
            True if at least 1 instance has a bounding box otherwise False.
        """
        if self.has_instances():
            return any([instance.has_bboxes() for instance in self.instances])
        return False

    def get_bboxes(self) -> torch.Tensor:
        """Get the bounding boxes of all instances in the frame.

        Returns:
            an (N,4) shaped tensor with bounding boxes of each instance in the frame.
        """
        if not self.has_instances():
            return torch.empty(0, 4)
        return torch.cat([instance.bbox for instance in self.instances], dim=0)

    def has_crops(self) -> bool:
        """Check if any of frames instances has a crop.

        Returns:
            True if at least 1 instance has a crop otherwise False.
        """
        if self.has_instances():
            return any([instance.has_crop() for instance in self.instances])
        return False

    def get_crops(self) -> torch.Tensor:
        """Get the crops of all instances in the frame.

        Returns:
            an (N, C, H, W) shaped tensor with crops of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        try:
            return torch.cat([instance.crop for instance in self.instances], dim=0)
        except Exception as e:
            print(self)
            raise (e)

    def has_features(self):
        """Check if any of frames instances has reid features already computed.

        Returns:
            True if at least 1 instance have reid features otherwise False.
        """
        if self.has_instances():
            return any([instance.has_features() for instance in self.instances])
        return False

    def get_features(self):
        """Get the reid feature vectors of all instances in the frame.

        Returns:
            an (N, D) shaped tensor with reid feature vectors of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.features for instance in self.instances], dim=0)

    def get_anchors(self) -> list[str]:
        """Get the anchor names of instances in the frame.

        Returns:
            A list of anchor names used by the instances to get the crop.
        """
        return [instance.anchor for instance in self.instances]

    def get_centroids(self) -> tuple[list[str], ArrayLike]:
        """Get the centroids around which each instance's crop was formed.

        Returns:
            anchors: the node names for the corresponding point
            points: an n_instances x 2 array containing the centroids
        """
        anchors = [
            anchor for instance in self.instances for anchor in instance.centroid.keys()
        ]

        points = np.array(
            [
                point
                for instance in self.instances
                for point in instance.centroid.values()
            ]
        )

        return (anchors, points)
