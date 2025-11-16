"""Module handling sliding window tracking."""

import logging
from collections import deque

import numpy as np
from torch import device
from dreem.io import Frame

logger = logging.getLogger("dreem.inference")


class TrackQueue:
    """Class handling track local queue system for sliding window.

    Each trajectory has its own deque based queue of size `window_size - 1`.
    Elements of the queue are Instance objects that have already been tracked
    and will be compared against later frames for assignment.
    """

    def __init__(
        self, window_size: int, max_gap: int = np.inf, verbose: bool = False
    ) -> None:
        """Initialize track queue.

        Args:
            window_size: The number of instances per trajectory allowed in the
                queue to be compared against.
            max_gap: The number of consecutive frames a trajectory can fail to
                appear in before terminating the track.
            verbose: Whether to print info during operations.
        """
        self._window_size = window_size
        self._queues = {}
        self._max_gap = max_gap
        self._curr_gap = {}
        if self._max_gap <= self._window_size:
            self._max_gap = self._window_size
        self._curr_track = set()
        self._verbose = verbose

    def __len__(self) -> int:
        """Get length of the queue.

        Returns:
            The total number of instances in every sub-queue.
        """
        return sum([len(queue) for queue in self._queues.values()])

    def __repr__(self) -> str:
        """Return the string representation of the TrackQueue.

        Returns:
            The string representation of the current state of the queue.
        """
        return (
            "TrackQueue("
            f"window_size={self.window_size}, "
            f"max_gap={self.max_gap}, "
            f"n_tracks={self.n_tracks}, "
            f"queues={[(key, len(queue)) for key, queue in self._queues.items()]}, "
            f"curr_gap:{self._curr_gap}"
            ")"
        )

    @property
    def window_size(self) -> int:
        """The maximum number of instances allowed in a sub-queue to be compared against.

        Returns:
            An int representing The maximum number of instances allowed in a
                sub-queue to be compared against.
        """
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int) -> None:
        """Set the window size of the queue.

        Args:
            window_size: An int representing The maximum number of instances
                allowed in a sub-queue to be compared against.
        """
        self._window_size = window_size

    @property
    def max_gap(self) -> int:
        """The maximum number of consecutive frames an trajectory can fail to appear before termination.

        Returns:
            An int representing the maximum number of consecutive frames an trajectory can fail to
                appear before termination.
        """
        return self._max_gap

    @max_gap.setter
    def max_gap(self, max_gap: int) -> None:
        """Set the max consecutive frame gap allowed for a trajectory.

        Args:
            max_gap: An int representing the maximum number of consecutive frames an trajectory can fail to
                appear before termination.
        """
        self._max_gap = max_gap

    @property
    def curr_track(self) -> int:
        """The newest *created* trajectory in the queue.

        Returns:
            The latest *created* trajectory in the queue.
        """
        return self._curr_track

    @curr_track.setter
    def curr_track(self, curr_track: int) -> None:
        """Set the newest *created* trajectory in the queue.

        Args:
            curr_track: The latest *created* trajectory in the queue.
        """
        self._curr_track = curr_track

    @property
    def n_tracks(self) -> int:
        """The current number of trajectories in the queue.

        Returns:
            An int representing the current number of trajectories in the queue.
        """
        return len(self._queues.keys())

    @property
    def tracks(self) -> list:
        """A list of the track ids currently in the queue.

        Returns:
            A list containing the track ids currently in the queue.
        """
        return list(self._queues.keys())

    @property
    def verbose(self) -> bool:
        """Indicate whether or not to print outputs along operations. Mostly used for debugging.

        Returns:
            A boolean representing whether or not printing is turned on.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool) -> None:
        """Turn on/off printing.

        Args:
            verbose: A boolean representing whether printing should be on or off.
        """
        self._verbose = verbose

    def end_tracks(self, track_id: int | None = None) -> bool:
        """Terminate tracks and removing them from the queue.

        Args:
            track_id: The index of the trajectory to be ended and removed.
                If `None` then then every trajectory is removed and the track queue is reset.

        Returns:
            True if the track is successively removed, otherwise False.
                (ie if the track doesn't exist in the queue.)
        """
        if track_id is None:
            # for track, instances in self._queues.items()
            self._queues = {}
            self._curr_gap = {}
            self.curr_track = set()
        else:
            try:
                self._queues.pop(track_id)
                self._curr_gap.pop(track_id)
            except KeyError:
                logger.exception(f"Track ID {track_id} not found in queue!")
                return False
        return True

    def add_frame(self, frame: Frame) -> dict[int, bool]:
        """Add frames to the queue.

        Each instance from the frame is added to the queue according to its pred_track_id.
        If the corresponding trajectory is not already in the queue then create a new queue for the track.

        Args:
            frame: A Frame object containing instances that have already been tracked.
        Returns:
            Dictionary mapping track ids to a bool indicating whether the track has exceeded the max gap and been terminated.
        """
        if frame.num_detected == 0:  # only add frames with instances.
            return
        vid_id = frame.video_id.item()
        frame_id = frame.frame_id.item()
        img_shape = frame.img_shape
        if isinstance(frame.video, str):
            vid_name = frame.video
        else:
            vid_name = frame.video.filename
        # traj_score = frame.get_traj_score()  TODO: figure out better way to save trajectory scores.
        frame_meta = (vid_id, frame_id, vid_name, list(img_shape))

        pred_tracks = []
        for instance in frame.instances:
            if instance.pred_track_id == -1:
                continue
            pred_track_id = instance.pred_track_id.item()
            pred_tracks.append(pred_track_id)

            if pred_track_id not in self._queues.keys():
                self._queues[pred_track_id] = deque(
                    [(*frame_meta, instance)]
                )  # dumb work around to retain `img_shape`
                self.curr_track.add(pred_track_id)

                logger.debug(
                    f"New track = {pred_track_id} on frame {frame_id}! Current number of tracks = {self.n_tracks}"
                )

            else:
                self._queues[pred_track_id].append((*frame_meta, instance))
            if len(self._queues[pred_track_id]) > self.window_size:
                popped = self._queues[pred_track_id].popleft()
                inst = popped[-1].to("cpu")
                del inst, popped
        gap = self.increment_gaps(
            pred_tracks
        )  # should this be done in the tracker or the queue?
        return gap

    def collate_tracks(
        self,
        device: str | device | None = None,
        track_ids: list[int] | None = None,
    ) -> list[Frame]:
        """Merge queues into a single list of Frames containing corresponding instances.

        Args:
            context_start_frame_id: The frame_id of the last frame in the context i.e. just before the start of the current batch
            track_ids: A list of trajectorys to merge. If None, then merge all
                queues, otherwise filter queues by track_ids then merge.

        Returns:
            A sorted list of Frame objects from which each instance came from,
            containing the corresponding instances.
        """
        if len(self._queues) == 0:
            return []

        frames = {}

        tracks_to_convert = (
            {track: queue for track, queue in self._queues if track in track_ids}
            if track_ids is not None
            else self._queues
        )
        for track, instances in tracks_to_convert.items():
            for video_id, frame_id, vid_name, img_shape, instance in instances:
                # if frame_id < context_start_frame_id - self.window_size:
                #     continue
                if (video_id, frame_id) not in frames.keys():
                    frame = Frame(
                        video_id,
                        frame_id,
                        img_shape=img_shape,
                        instances=[instance],
                        vid_file=vid_name,
                    )
                    frames[(video_id, frame_id)] = frame
                else:
                    frames[(video_id, frame_id)].instances.append(instance)
        return [frames[frame].to(device) for frame in sorted(frames.keys())]

    def increment_gaps(self, pred_track_ids: list[int]) -> dict[int, bool]:
        """Keep track of number of consecutive frames each trajectory has been missing from the queue.

        If a trajectory has exceeded the `max_gap` then terminate the track and remove it from the queue.

        Args:
            pred_track_ids: A list of track_ids to be matched against the trajectories in the queue.
                If a trajectory is in `pred_track_ids` then its gap counter is reset,
                otherwise its incremented by 1.

        Returns:
            A dictionary containing the trajectory id and a boolean value representing
            whether or not it has exceeded the max allowed gap and been
            terminated.
        """
        exceeded_gap = {}

        for track in pred_track_ids:
            if track not in self._curr_gap:
                self._curr_gap[track] = 0

        for track in self._curr_gap:
            if track not in pred_track_ids:
                self._curr_gap[track] += 1
                logger.debug(
                    f"Track {track} has not been seen for {self._curr_gap[track]} frames."
                )
            else:
                self._curr_gap[track] = 0
            if self._curr_gap[track] >= self.max_gap:
                exceeded_gap[track] = True
            else:
                exceeded_gap[track] = False

        for track, gap_exceeded in exceeded_gap.items():
            if gap_exceeded:
                logger.debug(
                    f"Track {track} has not been seen for {self._curr_gap[track]} frames! Terminating Track...Current number of tracks = {self.n_tracks}."
                )
                track_instances = self._queues.pop(track)
                for tup in track_instances:
                    inst = tup[-1]
                    inst = inst.to("cpu")
                del track_instances

                self._curr_gap.pop(track)

        return exceeded_gap
