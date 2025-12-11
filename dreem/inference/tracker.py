"""Module containing logic for going from association -> assignment."""

import logging
from math import inf, radians as deg2rad

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from dreem.datasets.data_utils import _pairwise_iou, is_pose_centroid_only
from dreem.inference.boxes import Boxes
from dreem.inference.post_processing import (
    ConfidenceFlagging,
    DistanceWeighting,
    IOUWeighting,
    OrientationWeighting,
)
from dreem.inference.post_processing_utils import get_principal_axis_with_fallback
from dreem.inference.track_queue import TrackQueue
from dreem.io import Frame
from dreem.models import GlobalTrackingTransformer, model_utils

logger = logging.getLogger("dreem.inference")


class Tracker:
    """Tracker class used for assignment based on sliding inference from GTR."""

    def __init__(
        self,
        window_size: int = 8,
        use_vis_feats: bool = True,
        overlap_thresh: float = 0.01,
        mult_thresh: bool = True,
        decay_time: float | None = None,
        iou: str | None = None,
        max_center_dist: float | None = None,
        distance_penalty_multiplier: float = 1.0,
        angle_diff_penalty_multiplier: float = 1.0,
        max_gap: int = inf,
        max_tracks: int = inf,
        verbose: bool = False,
        confidence_threshold: float = 0.0,
        temperature: float = 0.1,
        max_angle_diff: float = 0.0,
        front_nodes: list[str] | None = None,
        back_nodes: list[str] | None = None,
        enable_crop_saving: bool = False,
        **kwargs,
    ):
        """Initialize a tracker to run inference.

        Args:
            window_size: the size of the window used during sliding inference.
            use_vis_feats: Whether or not to use visual feature extractor.
            overlap_thresh: the trajectory overlap threshold to be used for assignment.
            mult_thresh: Whether or not to use weight threshold.
            decay_time: weight for `decay_time` postprocessing.
            iou: Either [None, '', "mult" or "max"]
                 Whether to use multiplicative or max iou reweighting.
            max_center_dist: distance threshold for filtering trajectory score matrix.
            max_gap: the max number of frames a trajectory can be missing before termination.
            max_tracks: the maximum number of tracks that can be created while tracking.
                We force the tracker to assign instances to a track instead of creating a new track if max_tracks has been reached.
            verbose: Whether or not to turn on debug printing after each operation.
            confidence_threshold: threshold for filtering out instances with high confidence. Set to 0 to disable confidence thresholding.
            temperature: temperature for softmax.
            max_angle_diff: maximum angle difference between pose principal axes when considering association between two instances. Set to 0 to disable angle difference filtering.
            front_nodes: list of skeleton node names to be used to determine the orientation of the object. If None, computes using all available nodes.
            back_nodes: list of skeleton node names to be used to determine the orientation of the object. If None, computes using all available nodes.
            enable_crop_saving: Whether to save crops to frame metadata.
            **kwargs: Additional keyword arguments (unused but accepted for compatibility).
        """
        self.track_queue = TrackQueue(
            window_size=window_size, max_gap=max_gap, verbose=verbose
        )
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist if max_center_dist is not None else inf
        self.verbose = verbose
        self.max_tracks = max_tracks
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.front_nodes = front_nodes
        self.back_nodes = back_nodes
        self.enable_crop_saving = enable_crop_saving
        self.max_angle_diff = (
            deg2rad(max_angle_diff) if max_angle_diff is not None else inf
        )
        self.orientation_weighting = OrientationWeighting(
            self.max_angle_diff, angle_diff_penalty_multiplier
        )
        self.distance_weighting = DistanceWeighting(
            self.max_center_dist, distance_penalty_multiplier
        )
        self.iou_weighting = IOUWeighting(iou)
        self.confidence_flagging = ConfidenceFlagging(confidence_threshold)

    def __call__(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> list[Frame]:
        """Wrap around `track` to enable `tracker()` instead of `tracker.track()`.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: list of Frames to run inference on

        Returns:
            List of frames containing association matrix scores and instances populated with pred track ids.
        """
        return self.track(model, frames)

    def __repr__(self) -> str:
        """Get string representation of tracker.

        Returns: the string representation of the tracker
        """
        return (
            "Tracker("
            f"max_tracks={self.max_tracks}, "
            f"use_vis_feats={self.use_vis_feats}, "
            f"overlap_thresh={self.overlap_thresh}, "
            f"mult_thresh={self.mult_thresh}, "
            f"decay_time={self.decay_time}, "
            f"max_center_dist={self.max_center_dist}, "
            f"verbose={self.verbose}, "
            f"queue={self.track_queue}, "
            f"temperature={self.temperature}"
            f"queue={self.track_queue}, "
            f"temperature={self.temperature}"
        )

    def track(
        self, model: GlobalTrackingTransformer, frames: list[dict]
    ) -> list[Frame]:
        """Run tracker and get predicted trajectories.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: data dict to run inference on

        Returns:
            List of Frames populated with pred track ids and association matrix scores
        """
        _ = model.eval()
        instances_pred = self.sliding_inference(model, frames)
        return instances_pred

    def sliding_inference(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> list[Frame]:
        """Perform sliding inference on the input video (instances) with a given window size.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames (See `dreem.io.Frame` for more info).

        Returns:
            frames: A list of Frames populated with pred_track_ids and asso_matrices
        """
        # B: batch size.
        # D: embedding dimension.
        # nc: number of channels.
        # H: height.
        # W: width.
        for batch_idx, frame_to_track in enumerate(frames):
            tracked_frames = self.track_queue.collate_tracks(
                device=frame_to_track.device
            )
            logger.debug(f"Current number of tracks is {self.track_queue.n_tracks}")
            if frame_to_track.frame_id == 0:  # clear queue on new video
                logger.debug("New Video! Resetting Track Queue.")
                self.track_queue.end_tracks()

            # Initialize tracks on first frame where detections appear
            if len(self.track_queue) == 0:
                if frame_to_track.has_instances():
                    logger.debug(
                        f"Initializing track on clip ind {batch_idx} frame {frame_to_track.frame_id.item()}"
                    )

                    curr_track_id = 0
                    for i, instance in enumerate(frames[batch_idx].instances):
                        instance.pred_track_id = instance.gt_track_id
                        curr_track_id = max(curr_track_id, instance.pred_track_id)

                    for i, instance in enumerate(frames[batch_idx].instances):
                        if instance.pred_track_id == -1:
                            instance.pred_track_id = curr_track_id
                            curr_track_id += 1

            else:
                if frame_to_track.has_instances():  # Check if there are detections. If there are skip and increment gap count
                    frames_to_track = tracked_frames + [
                        frame_to_track
                    ]  # better var name?
                    query_ind = len(frames_to_track) - 1
                    frame_to_track = self._run_global_tracker(
                        model,
                        frames_to_track,
                        query_ind=query_ind,
                    )
                    del frames_to_track

            if frame_to_track.has_instances():
                _ = self.track_queue.add_frame(frame_to_track)
            else:
                self.track_queue.increment_gaps([])

            frames[batch_idx] = frame_to_track

        del frame_to_track, tracked_frames
        torch.cuda.empty_cache()
        return frames

    def _run_global_tracker(
        self, model: GlobalTrackingTransformer, frames: list[Frame], query_ind: int
    ) -> Frame:
        """Run global tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames containing reid features. See `dreem.io.data_structures` for more info.
            query_ind: An integer for the query frame within the window of instances.

        Returns:
            query_frame: The query frame now populated with the pred_track_ids.
        """
        # *: each item in frames is a frame in the window. So it follows
        #    that each frame in the window has * detected instances.
        # D: embedding dimension.
        # total_instances: number of instances in the window.
        # N_i: number of detected instances in i-th frame of window.
        # instances_per_frame: a list of number of instances in each frame of the window.
        # n_query: number of instances in current/query frame (rightmost frame of the window).
        # n_nonquery: number of instances in the window excluding the current/query frame.
        # window_size: length of window.
        # L: number of decoder blocks.
        # n_traj: number of existing tracks within the window so far.

        # Number of instances in each frame of the window.
        # E.g.: instances_per_frame: [4, 5, 6, 7]; window of length 4 with 4 detected instances in the first frame of the window.

        _ = model.eval()

        query_frame = frames[query_ind]
        _, h, w = query_frame.img_shape

        query_instances = query_frame.instances
        all_instances = [instance for frame in frames for instance in frame.instances]

        logger.debug(f"Frame {query_frame.frame_id.item()}")

        instances_per_frame = [frame.num_detected for frame in frames]

        total_instances = sum(instances_per_frame)  # Number of instances in window

        logger.debug(f"total_instances: {total_instances}")

        overlap_thresh = self.overlap_thresh
        mult_thresh = self.mult_thresh
        n_traj = self.track_queue.n_tracks
        curr_tracks = self.track_queue.curr_track

        query_poses = [
            (instance.pose, instance.pred_track_id.item(), instance.crop.clone().cpu())
            for instance in query_frame.instances
        ]
        nonquery_poses = [
            (instance.pose, instance.pred_track_id.item(), instance.crop.clone().cpu())
            for nonquery_frame in frames
            if nonquery_frame.frame_id != query_frame.frame_id.item()
            for instance in nonquery_frame.instances
        ]
        enable_crop_saving = self.enable_crop_saving or is_pose_centroid_only(
            query_poses[0][0]
        )
        if (
            is_pose_centroid_only(query_poses[0][0])
            and query_frame.frame_id.item() == 1
            and self.max_angle_diff > 0
        ):
            logger.warning(
                "Crop saving is enabled due to max_angle_diff > 0 and a centroid-only skeleton. This can increase memory usage. If you experience memory issues, consider setting max_angle_diff = 0. This will skip orientation based post processing."
            )
        with torch.no_grad():
            asso_matrix = model(
                all_instances, query_instances, retain_crops=enable_crop_saving
            )

        asso_output = asso_matrix[-1].matrix.split(
            instances_per_frame, dim=1
        )  # (window_size, n_query, N_i)
        asso_output = model_utils.softmax_asso(
            asso_output
        )  # (window_size, n_query, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (n_query, total_instances)

        asso_output_df = pd.DataFrame(
            asso_output.clone().numpy(),
            columns=[f"Instance {i}" for i in range(asso_output.shape[-1])],
        )

        asso_output_df.index.name = "Instances"
        asso_output_df.columns.name = "Instances"

        query_frame.add_traj_score("asso_output", asso_output_df)
        query_frame.asso_output = asso_matrix[-1]

        n_query = (
            query_frame.num_detected
        )  # Number of instances in the current/query frame.

        n_nonquery = (
            total_instances - n_query
        )  # Number of instances in the window not including the current/query frame.

        logger.debug(f"n_nonquery: {n_nonquery}")
        logger.debug(f"n_query: {n_query}")

        instance_ids = (
            torch.cat(
                [
                    x.get_pred_track_ids()
                    for batch_idx, x in enumerate(frames)
                    if batch_idx != query_ind
                ],
                dim=0,
            )
            .view(n_nonquery)
            .cpu()
        )  # (n_nonquery,)

        query_inds = [
            x
            for x in range(
                sum(instances_per_frame[:query_ind]),
                sum(instances_per_frame[: query_ind + 1]),
            )
        ]

        nonquery_inds = [i for i in range(total_instances) if i not in query_inds]

        # instead should we do model(nonquery_instances, query_instances)?
        asso_nonquery = asso_output[:, nonquery_inds]  # (n_query, n_nonquery)

        asso_nonquery_df = pd.DataFrame(
            asso_nonquery.clone().numpy(), columns=nonquery_inds
        )

        asso_nonquery_df.index.name = "Current Frame Instances"
        asso_nonquery_df.columns.name = "Nonquery Instances"

        query_frame.add_traj_score("asso_nonquery", asso_nonquery_df)

        # get raw bbox coords of prev frame instances from frame.instances_per_frame
        query_boxes_px = torch.cat(
            [instance.bbox for instance in query_frame.instances], dim=0
        ).cpu()

        pred_boxes = model_utils.get_boxes(all_instances)
        query_boxes = pred_boxes[query_inds]  # n_k x 4
        nonquery_boxes = pred_boxes[nonquery_inds]  # n_nonquery x 4

        unique_ids = torch.unique(instance_ids).cpu()  # (n_nonquery,)

        logger.debug(f"Instance IDs: {instance_ids}")
        logger.debug(f"unique ids: {unique_ids}")

        id_inds = (
            unique_ids[None, :] == instance_ids[:, None]
        ).float()  # (n_nonquery, n_traj)

        ################################################################################

        # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_query x n_traj
        traj_score = torch.mm(asso_nonquery, id_inds.cpu())  # (n_query, n_traj)
        assoc_col_pred_id_map = {
            i: unique_ids[i].item() for i in range(unique_ids.shape[0])
        }
        traj_score_df = pd.DataFrame(
            traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
        )

        traj_score_df.index.name = "Current Frame Instances"
        traj_score_df.columns.name = "Unique IDs"

        query_frame.add_traj_score("traj_score", traj_score_df)
        ################################################################################

        # with iou -> combining with location in tracker, they set to True
        # todo -> should also work without pos_embed

        if id_inds.numel() > 0:
            # this throws error, think we need to slice?
            # last_inds = (id_inds * torch.arange(
            #    n_nonquery, device=id_inds.device)[:, None]).max(dim=0)[1] # n_traj

            last_inds = (id_inds * torch.arange(n_nonquery)[:, None]).max(dim=0)[1]  # M

            last_boxes = nonquery_boxes[last_inds]  # n_traj x 4
            last_ious = _pairwise_iou(Boxes(query_boxes), Boxes(last_boxes))  # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)

        state = self.iou_weighting.run(
            {
                "traj_score": traj_score,
                "last_ious": last_ious.cpu(),
            }
        )
        traj_score = state["traj_score"]

        if self.iou is not None and self.iou != "":
            iou_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
            )

            iou_traj_score.index.name = "Current Frame Instances"
            iou_traj_score.columns.name = "Unique IDs"

            query_frame.add_traj_score("weight_iou", iou_traj_score)
        ################################################################################
        last_poses = []
        for ind in last_inds.cpu():  # its important to index nonquery_poses by last_inds as this maintains ordering that matches the association matrix ordering
            last_poses.append(nonquery_poses[ind])
        last_pred_ids = [pred_track_id for _, pred_track_id, _ in last_poses]

        if self.max_center_dist is not None and self.max_center_dist > 0:
            last_boxes_px = last_boxes.cpu()
            last_boxes_px[:, :, [0, 2]] *= w
            last_boxes_px[:, :, [1, 3]] *= h
            state = self.distance_weighting.run(
                {
                    "traj_score": traj_score,
                    "query_boxes_px": query_boxes_px,
                    "last_boxes_px": last_boxes_px,
                    "h": h,
                    "w": w,
                }
            )
            traj_score = state["traj_score"]
            # update metadata
            max_center_dist_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
            )

            max_center_dist_traj_score.index.name = "Current Frame Instances"
            max_center_dist_traj_score.columns.name = "Unique IDs"

            query_frame.add_traj_score("max_center_dist", max_center_dist_traj_score)

        ################################################################################

        if self.max_angle_diff > 0:
            query_principal_axes = []
            last_principal_axes = []
            successes = []
            for instance, pred_track_id, crop in query_poses:
                instance_principal_axes, success = get_principal_axis_with_fallback(
                    instance,
                    self.front_nodes,
                    self.back_nodes,
                    crop,
                    logger,
                    query_frame.frame_id.item(),
                )
                query_principal_axes.append(instance_principal_axes)
                successes.append(success)
            query_principal_axes = torch.stack(query_principal_axes)  # (n_query, 2)

            for instance, pred_track_id, crop in last_poses:
                instance_principal_axes, success = get_principal_axis_with_fallback(
                    instance,
                    self.front_nodes,
                    self.back_nodes,
                    crop,
                    logger,
                    query_frame.frame_id.item(),
                )
                last_principal_axes.append(instance_principal_axes)
                successes.append(success)
            last_principal_axes = torch.stack(last_principal_axes)  # (n_traj, 2)

            if all(successes):
                state = self.orientation_weighting.run(
                    {
                        "traj_score": traj_score,
                        "query_principal_axes": query_principal_axes,
                        "last_principal_axes": last_principal_axes,
                    }
                )
                traj_score = state["traj_score"]
                # update metadata
                angle_diff_weight_traj_score = pd.DataFrame(
                    traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
                )
                angle_diff_weight_traj_score.index.name = "Current Frame Instances"
                angle_diff_weight_traj_score.columns.name = "Unique IDs"

                query_frame.add_traj_score(
                    "angle_diff_weight", angle_diff_weight_traj_score
                )
            else:
                logger.warning(
                    f"Some instances had no principal axis in frame {query_frame.frame_id.item()}. Skipping instance angle based post processing."
                )

        ################################################################################
        scaled_traj_score = torch.nn.functional.log_softmax(
            traj_score / self.temperature, dim=1
        )
        scaled_traj_score_df = pd.DataFrame(
            scaled_traj_score.numpy(), columns=unique_ids.cpu().numpy()
        )
        scaled_traj_score_df.index.name = "Current Frame Instances"
        scaled_traj_score_df.columns.name = "Unique IDs"

        query_frame.add_traj_score("scaled", scaled_traj_score_df)
        ################################################################################
        # Flag frames with low confidence
        if self.confidence_threshold > 0:
            state = self.confidence_flagging.run(
                {
                    "scaled_traj_score": scaled_traj_score,
                    "n_query": n_query,
                    "query_frame": query_frame,
                }
            )
            scaled_traj_score = state["scaled_traj_score"]

        match_i, match_j = linear_sum_assignment((-traj_score))

        track_ids = instance_ids.new_full((n_query,), -1)
        for i, j in zip(match_i, match_j):
            # The overlap threshold is multiplied by the number of times the unique track j is matched to an
            # instance out of all instances in the window excluding the current frame.
            #
            # So if this is correct, the threshold is higher for matching an instance from the current frame
            # to an existing track if that track has already been matched several times.
            # So if an existing track in the window has been matched a lot, it gets harder to match to that track.
            thresh = (
                overlap_thresh * id_inds[:, j].sum() if mult_thresh else overlap_thresh
            )
            # if we've already reached max tracks, just assign it to one of the existing tracks
            if n_traj >= self.max_tracks or traj_score[i, j] > thresh:
                logger.debug(
                    f"Assigning instance {i} to track {j} with id {unique_ids[j]}"
                )
                track_ids[i] = unique_ids[j]
                query_frame.instances[i].track_score = scaled_traj_score[i, j].item()
        logger.debug(f"track_ids: {track_ids}")
        for i in range(n_query):
            # True if association score was below the threshold, and we haven't reached max tracks
            if track_ids[i] < 0 and n_traj < self.max_tracks:
                max_track_id = max(curr_tracks)
                logger.debug(f"Creating new track {max_track_id + 1}")
                curr_tracks.add(max_track_id + 1)
                track_ids[i] = max_track_id + 1

        query_frame.matches = (match_i, match_j)

        for instance, track_id in zip(query_frame.instances, track_ids):
            instance.pred_track_id = track_id

        final_traj_score = pd.DataFrame(
            traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
        )
        final_traj_score.index.name = "Current Frame Instances"
        final_traj_score.columns.name = "Unique IDs"

        query_frame.add_traj_score("final", final_traj_score)

        return query_frame
