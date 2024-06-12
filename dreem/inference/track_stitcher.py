"""Module containing logic for associating instances by track (tracklet stitching).

Note this is an experimental module!
"""

import torch
import pandas as pd
import logging

from dreem.io import Frame, AssociationMatrix
from dreem.models import model_utils, GlobalTrackingTransformer
from dreem.inference.track_queue import TrackQueue
from dreem.inference import post_processing
from dreem.inference.boxes import Boxes
from scipy.optimize import linear_sum_assignment
from math import inf

logger = logging.getLogger("dreem.inference")


class TrackStitcher:
    """Track stitching class used for assignment by stitching tracklets together."""

    def __init__(
        self,
        ref_tracklet_length: int = 8,
        query_tracklet_length: int = 8,
        use_vis_feats: bool = True,
        decay_time: float | None = None,
        iou: str | None = None,
        max_center_dist: float | None = None,
        persistent_tracking: bool = False,
        verbose: bool = False,
    ):
        """Initialize a tracker to run inference.

        Args:
            ref_tracklet_length: the number of frames used to get reference tracklets to match to.
            query_tracklet_length: the number of frames used to get query tracklets to stitch to reference.
            use_vis_feats: Whether or not to use visual feature extractor.
            decay_time: weight for `decay_time` postprocessing.
            iou: Either [None, '', "mult" or "max"]
                 Whether to use multiplicative or max iou reweighting.
            max_center_dist: distance threshold for filtering trajectory score matrix.
            persistent_tracking: whether to keep a buffer across chunks or not.
        """
        self.track_queue = TrackQueue(window_size=ref_tracklet_length, verbose=verbose)
        self.query_tracklet_length = query_tracklet_length
        self.use_vis_feats = use_vis_feats
        self.decay_time = decay_time
        self.iou = None  # turned off for now -> need to adapt
        self.max_center_dist = None  # turned off for now -> need to adapt
        self.persistent_tracking = persistent_tracking

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
        return self.stitch(model, frames)

    def stitch(
        self, model: GlobalTrackingTransformer, frames: list[dict]
    ) -> list[Frame]:
        """Run tracker and get predicted trajectories.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: data dict to run inference on

        Returns:
            List of Frames populated with pred track ids and association matrix scores
        """
        # Extract feature representations with pre-trained encoder.

        _ = model.eval()

        for frame in frames:
            if frame.has_instances():
                if not self.use_vis_feats:
                    for instance in frame.instances:
                        instance.features = torch.zeros(1, model.d_model)
                    # frame["features"] = torch.randn(
                    #     num_frame_instances, self.model.d_model
                    # )

                # comment out to turn encoder off

                # Assuming the encoder is already trained or train encoder jointly.
                elif not frame.has_features():
                    with torch.no_grad():
                        crops = frame.get_crops()
                        z = model.visual_encoder(crops)

                        for i, z_i in enumerate(z):
                            frame.instances[i].features = z_i

        # I feel like this chunk is unnecessary:
        # reid_features = torch.cat(
        #     [frame["features"] for frame in instances], dim=0
        # ).unsqueeze(0)

        # asso_preds, pred_boxes, pred_time, embeddings = self.model(
        #     instances, reid_features
        # )
        instances_pred = self.sliding_inference(model, frames)

        if not self.persistent_tracking:
            logger.debug(f"Clearing Queue after tracking")
            self.track_queue.end_tracks()

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

        for batch_idx in range(0, len(frames), self.query_tracklet_length):
            logger.debug(f"Current number of tracks is {self.track_queue.n_tracks}")
            frames_to_track = pred_tracklets = frames[
                batch_idx : self.query_tracklet_length
            ]
            if (
                self.persistent_tracking and frames[batch_idx].frame_id == 0
            ):  # check for new video and clear queue

                logger.debug("New Video! Resetting Track Queue.")
                self.track_queue.end_tracks()

            """
            Initialize tracks on first frame where detections appear.
            """
            if len(self.track_queue) == 0:
                for frame in pred_tracklets:
                    if frame.has_instances():

                        logger.debug(
                            f"Initializing tracklet on clip ind {batch_idx} frame {frame.frame_id.item()}"
                        )

                        for i, instance in enumerate(frame.instances):
                            instance.pred_track_id = instance.gt_track_id

            else:
                tracked_frames = self.track_queue.collate_tracks()
                if any([frame.has_instances() for frame in frames_to_track]):
                    pred_tracklets = self._run_global_tracker(
                        model,
                        tracked_frames,
                        frames_to_track,
                    )
            for frame in pred_tracklets:
                if frame.has_instances():
                    self.track_queue.add_frame(frame)
                else:
                    self.track_queue.increment_gaps([])

            frames[batch_idx : batch_idx + self.query_tracklet_length] = pred_tracklets
        return frames

    def _run_global_tracker(
        self,
        model: GlobalTrackingTransformer,
        tracked_frames: list[Frame],
        frames_to_track: list[Frame],
    ) -> list[Frame]:
        """Run global tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            tracked_frames: A list of Frames containing already tracked instances. See `dreem.io.data_structures` for more info.
            frames_to_track: A list of Frames containing instances to be prediced on. See `dreem.io.data_structures` for more info.

        Returns:
            pred_tracklets: The query frame now populated with the pred_track_ids.
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

        all_frames = tracked_frames + frames_to_track

        query_instances = [
            instance for frame in frames_to_track for instance in frame.instances
        ]
        ref_instances = [
            instance for frame in tracked_frames for instance in frame.instances
        ]
        all_instances = [
            instance for frame in all_frames for instance in frame.instances
        ]

        instances_per_frame = [frame.num_detected for frame in all_frames]

        total_instances, window_size = sum(instances_per_frame), len(
            instances_per_frame
        )  # Number of instances in window; length of window.

        logger.debug(f"total_instances: {total_instances}")

        reid_features = torch.cat(
            [frame.get_features() for frame in all_frames], dim=0
        )[
            None
        ]  # (1, total_instances, D=512)

        # (L=1, n_query, total_instances)
        with torch.no_grad():
            asso_matrix = model(all_instances, query_instances)

        asso_output = asso_matrix[-1].matrix.split(
            instances_per_frame, dim=1
        )  # (window_size, n_query, N_i)
        asso_output = model_utils.softmax_asso(
            asso_output
        )  # (window_size, n_query, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (n_query, total_instances)

        n_query = len(
            query_instances
        )  # Number of instances in the current/query frame.

        n_nonquery = (
            total_instances - n_query
        )  # Number of instances in the window not including the current/query frame.

        logger.debug(f"n_nonquery: {n_nonquery}")
        logger.debug(f"n_query: {n_query}")

        # instead should we do model(nonquery_instances, query_instances)?
        asso_nonquery = asso_output[
            query_instances, ref_instances
        ]  # (n_query, n_nonquery)

        asso_nonquery = AssociationMatrix(
            asso_nonquery, ref_instances=ref_instances, query_instances=query_instances
        )

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        traj_score = asso_nonquery
        traj_score.matrix = post_processing.weight_decay_time(
            traj_score.matrix,
            self.decay_time,
            reid_features,
            window_size,
            len(all_frames),
        )

        ################################################################################

        # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_traj x n_traj
        traj_score = traj_score.reduce(
            "traj", "traj", "pred", "gt"
        )  # (n_query, n_traj)

        ################################################################################

        # with iou -> combining with location in tracker, they set to True
        # TODO adapt for traj x traj
        #
        # query_boxes = model_utils.get_boxes(query_instances)  # n_k x 4
        # nonquery_boxes = model_utils.get_boxes(ref_instances)  # n_nonquery x 4
        #
        # if id_inds.numel() > 0:
        #     # this throws error, think we need to slice?
        #     # last_inds = (id_inds * torch.arange(
        #     #    n_nonquery, device=id_inds.device)[:, None]).max(dim=0)[1] # n_traj

        #     last_inds = (
        #         id_inds * torch.arange(n_nonquery, device=id_inds.device)[:, None]
        #     ).max(dim=0)[
        #         1
        #     ]  # M

        #     last_boxes = nonquery_boxes[last_inds]  # n_traj x 4
        #     last_ious = post_processing._pairwise_iou(
        #         Boxes(query_boxes), Boxes(last_boxes)
        #     )  # n_k x M
        # else:
        #     last_ious = traj_score.new_zeros(traj_score.shape)

        # traj_score = post_processing.weight_iou(
        #     traj_score, self.iou, last_ious.cpu()
        # )

        ################################################################################

        # threshold for continuing a tracking or starting a new track -> they use 1.0
        # TODO -> implement for traj x traj
        # traj_score = post_processing.filter_max_center_dist(
        #     traj_score, self.max_center_dist, query_boxes, nonquery_boxes, id_inds
        # )

        ################################################################################

        # scale traj score via softmax for interpretability
        scaled_traj_score = torch.softmax(torch.tensor(traj_score.numpy()), dim=1)
        ################################################################################

        match_i, match_j = linear_sum_assignment((-traj_score))
        match_dict = {
            traj_score.index[i]: traj_score.columns[j] for i, j in zip(match_i, match_j)
        }

        for frame in frames_to_track:
            for instance in frame.instances:
                tracklet_id = instance.gt_track_id.item()
                instance.track_score = scaled_traj_score[
                    tracklet_id, match_dict[tracklet_id]
                ]
                instance.pred_track_id = match_dict[tracklet_id]

        return frames_to_track
