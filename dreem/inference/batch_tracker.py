"""Module containing logic for going from association -> assignment."""

import torch
import pandas as pd
import logging

from dreem.io import Frame
from dreem.models import model_utils, GlobalTrackingTransformer
from dreem.inference.track_queue import TrackQueue
from dreem.inference import post_processing
from dreem.inference.boxes import Boxes
from scipy.optimize import linear_sum_assignment
from math import inf

logger = logging.getLogger("dreem.inference")


class BatchTracker:
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
        persistent_tracking: bool = True,
        max_gap: int = inf,
        max_tracks: int = inf,
        verbose: bool = False,
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
            persistent_tracking: whether to keep a buffer across chunks or not.
            max_gap: the max number of frames a trajectory can be missing before termination.
            max_tracks: the maximum number of tracks that can be created while tracking.
                We force the tracker to assign instances to a track instead of creating a new track if max_tracks has been reached.
            verbose: Whether or not to turn on debug printing after each operation.
        """
        self.track_queue = TrackQueue(
            window_size=window_size, max_gap=max_gap, verbose=verbose
        )
        self.num_frames_tracked = 0
        self.window_size = window_size
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist
        self.persistent_tracking = persistent_tracking
        self.verbose = verbose
        self.max_tracks = max_tracks

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

        instances_pred = self.track(model, frames)
        # no more persistent tracking. It is on by default
        # if not self.persistent_tracking:
        #     logger.debug(f"Clearing Queue after tracking")
        #     self.track_queue.end_tracks()

        return instances_pred

    def __repr__(self) -> str:
        """Get string representation of tracker.

        Returns: the string representation of the tracker
        """
        return (
            "Tracker("
            f"persistent_tracking={self.persistent_tracking}, "
            f"max_tracks={self.max_tracks}, "
            f"use_vis_feats={self.use_vis_feats}, "
            f"overlap_thresh={self.overlap_thresh}, "
            f"mult_thresh={self.mult_thresh}, "
            f"decay_time={self.decay_time}, "
            f"max_center_dist={self.max_center_dist}, "
            f"verbose={self.verbose}, "
            f"queue={self.track_queue}"
        )

    def track(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> list[Frame]:
        """Perform sliding inference on the input video (instances) with a given window size. This method is called once per batch.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames (See `dreem.io.Frame` for more info).
        Returns:
            frames: A list of Frames populated with pred_track_ids and asso_matrices
        """
        # all batches up until context_length number of frames have been tracked, will be tracked frame-by-frame
        if self.num_frames_tracked < self.window_size:
            frames = self.track_by_frame(model, frames)
        else:
            frames = self.track_by_batch(model, frames)

        return frames


    def track_by_batch(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> list[Frame]:
        """Perform sliding inference, on an entire batch of frames, on the input video (instances) with a given context length (window size).

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames (See `dreem.io.Frame` for more info).
        Returns:
            frames: A list of Frames populated with pred_track_ids and asso_matrices
        """
        # context window starts from last frame just before start of current batch, to window_size frames preceding it
        # note; can't use last frame of previous batch, because there could be empty frames in between batches that must 
        # be part of the context window for consistency
        context_window_frames = self.track_queue.collate_tracks(
            context_start_frame_id=frames[0].frame_id.item() - 1, # switched off in collate_tracks; there is no cutoff for context, only until the deque gets filled
            device=frames[0].frame_id.device
        )

        context_window_instances = []
        context_window_instance_frame_ids = []
        for frame in context_window_frames:
            context_window_instances.extend(frame.instances)
            context_window_instance_frame_ids.extend([frame.frame_id] * len(frame.instances))
            
        current_batch_instances = []
        current_batch_instance_frame_ids = []
        for frame in frames:
            current_batch_instances.extend(frame.instances)
            current_batch_instance_frame_ids.extend([frame.frame_id] * len(frame.instances))
        
        frames_to_track = context_window_frames + frames

        # query is current batch instances, key is context window and current batch instances
        association_matrix = model(context_window_instances + current_batch_instances, current_batch_instances)

        # take association matrix and all frames off GPU (frames include instances)
        association_matrix = association_matrix[-1].to("cpu")
        context_window_frames = [frame.to("cpu") for frame in context_window_frames]
        frames = [frame.to("cpu") for frame in frames]

        # keep current batch instances in assoc matrix, and remove them after softmax (mirrors the training scheme)
        pred_frames = self._run_batch_tracker(association_matrix.matrix, context_window_frames, frames, compute_probs_by_frame=True)

        return pred_frames
    

    def track_by_frame(
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
            # if we're tracking by frame, it means context length of frames hasn't been reached yet, so context start frame id is 0
            context_window_frames = self.track_queue.collate_tracks(
                context_start_frame_id=0,
                device=frame_to_track.frame_id.device
            )
            logger.debug(f"Current number of tracks is {self.track_queue.n_tracks}")

            if (
                self.persistent_tracking and frame_to_track.frame_id == 0
            ):  # check for new video and clear queue

                logger.debug("New Video! Resetting Track Queue.")
                self.track_queue.end_tracks()

            """
            Initialize tracks on first frame where detections appear. This is the first frame of the first batch
            """
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
                            curr_track_id += 1
                            instance.pred_track_id = curr_track_id

            else:
                if (
                    frame_to_track.has_instances()
                ):  # Check if there are detections. If there are skip and increment gap count
                    frames_to_track = context_window_frames + [
                        frame_to_track
                    ]  # better var name?

                    query_ind = len(frames_to_track) - 1

                    frame_to_track = self._run_frame_by_frame_tracker(
                        model,
                        frames_to_track,
                    )

            if frame_to_track.has_instances():
                self.track_queue.add_frame(frame_to_track)
                self.num_frames_tracked += 1
            else:
                self.track_queue.increment_gaps([])

            frames[batch_idx] = frame_to_track

        return frames


    def _run_batch_tracker(
            self, association_matrix: torch.Tensor, context_window_frames: list[Frame], frames: list[Frame], compute_probs_by_frame: bool = True
    ) -> Frame:
        """
        Run batch tracker performs track assignment for each frame in the current batch. Supports 2 methods for computing association probabilities.
        First is to softmax each query instance in each query frame in the batch, with only 1 frame at a time from the context window. This is the default method
        and only supports local track linking.
        Second is to softmax the entire context + curr batch, then index. This enables global track linking via e.g. ILP. 
        In this case, prob values will be smaller and the overlap thresh should be decreased

        Args:
            association_matrix: the association matrix to be used for tracking
            :
            batch_start_ind: The index (in frames_to_track) of the first frame in the current batch
            compute_probs_by_frame: Whether to softmax the association matrix logits for each frame in context separately, or globally for the entire context window + current batch
        Returns:
            List of frames populated with pred_track_ids and asso_matrices
        """
        tracked_frames = []
        num_instances_per_frame = [frame.num_detected for frame in context_window_frames + frames]
        all_frames = context_window_frames + frames
        batch_start_ind = len(num_instances_per_frame) - len(frames)
        overlap_thresh = self.overlap_thresh
        mult_thresh = self.mult_thresh
        n_traj = self.track_queue.n_tracks
        curr_track = self.track_queue.curr_track

        for query_frame_idx, frame in enumerate(frames): # only track frames in current batch, not in context window
            all_prev_instances = [instance for frame in context_window_frames + frames[:query_frame_idx] for instance in frame.instances]
            # indices that will be used to index the rows of the association matrix corresponding to the query frame instances
            query_inds = [
                x
                for x in range(
                    sum(num_instances_per_frame[batch_start_ind:batch_start_ind + query_frame_idx]),
                    sum(num_instances_per_frame[batch_start_ind:batch_start_ind + query_frame_idx + 1]),
                )
            ]
            # first, slice the association matrix to only include the query frame instances along the rows; these are the 'detections' to be matched to tracks
            # recall incoming association_matrix is (num_instances_in_batch, num_instances_in_context_window + num_instances_in_batch)
            assoc_curr_frame = association_matrix[query_inds, :]
            # discard the columns (ref instances) corresponding to frames including and after the current frame; this means each frame will see previous frames in the batch as well as the context window when linking to tracks
            # importantly, this means that tracks will be aggregated over a much longer time period than the context window size, making many more tracks visible to each frame to link detections to
            assoc_curr_frame_by_previous_frames = assoc_curr_frame[:, :sum(num_instances_per_frame[:batch_start_ind + query_frame_idx])] # (num_query_instances, num instances in context window + num instances in current batch up till current frame)
            
            # method 1
            if compute_probs_by_frame:
                # for each frame in the context window, split the assoc matrix columns by frame
                split_assos = assoc_curr_frame_by_previous_frames.split(num_instances_per_frame[:batch_start_ind + query_frame_idx], dim=1)
                # compute softmax per-frame
                softmaxed_asso = model_utils.softmax_asso(split_assos)
                # merge the softmaxed assoc matrices back together to get (num_query_instances, num_instances_in_context_window)
                softmaxed_asso = torch.cat(softmaxed_asso, dim=1)
            # method 2
            else:
                # compute softmax across the entire context window and current batch frames
                softmaxed_asso = model_utils.softmax_asso(assoc_curr_frame_by_previous_frames)[0]

            # proceed with post processing, LSA, and track assignment (duplicated code with frame by frame tracker)

            # get raw bbox coords of prev frame instances from frame.instances_per_frame
            prev_frame = all_frames[batch_start_ind + query_frame_idx - 1]
            prev_frame_instance_ids = torch.cat(
                [instance.pred_track_id for instance in prev_frame.instances], dim=0
            )
            prev_frame_boxes = torch.cat(
                [instance.bbox for instance in prev_frame.instances], dim=0
            )
            all_prev_frames_boxes = torch.cat([instance.bbox for instance in all_prev_instances], dim=0)
            curr_frame_boxes = torch.cat(
                [instance.bbox for instance in frame.instances], dim=0
            )
            # get the pred track ids for all instances up until the current frame
            instance_ids = torch.cat(
            [
                x.get_pred_track_ids()
                for x in all_frames[:batch_start_ind + query_frame_idx]
            ],
            dim=0,
            ).view(
                sum(num_instances_per_frame[:batch_start_ind + query_frame_idx])
            )  # (n_nonquery,)

            unique_ids = torch.unique(instance_ids)  # (n_nonquery,)

            _, h, w = frame.img_shape.flatten()
            bbox_scaler = torch.tensor([w, h, w, h])
            query_boxes =  curr_frame_boxes / bbox_scaler # n_k x 4
            nonquery_boxes = all_prev_frames_boxes / bbox_scaler # n_nonquery x 4

            logger.debug(f"Instance IDs: {instance_ids}")
            logger.debug(f"unique ids: {unique_ids}")

            id_inds = (
                unique_ids[None, :] == instance_ids[:, None]
            ).float()  # (n_nonquery, n_traj)

            prev_frame_id_inds = (
                unique_ids[None, :] == prev_frame_instance_ids[:, None]
            ).float()  # (n_prev_frame_instances, n_traj)

            # aggregate the association matrix by tracks; output is shape (num_query_instances, num_tracks)
            # note the reduce operation is over the context window instances as well as current batch instances up until the current frame
            # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_query x n_traj
            traj_score = torch.mm(softmaxed_asso, id_inds.cpu())  # (n_query, n_traj)

            # post processing
            if id_inds.numel() > 0:
                last_inds = (
                    id_inds * torch.arange(len(all_prev_instances), device=id_inds.device)[:, None]
                ).max(dim=0)[
                    1
                ]  # M

                last_boxes = nonquery_boxes[last_inds]  # n_traj x 4
                last_ious = post_processing._pairwise_iou(
                    Boxes(query_boxes), Boxes(last_boxes)
                )  # n_k x M
            else:
                last_ious = traj_score.new_zeros(traj_score.shape)

            traj_score = post_processing.weight_iou(traj_score, self.iou, last_ious)

            # threshold for continuing a tracking or starting a new track -> they use 1.0
            traj_score = post_processing.filter_max_center_dist(
                traj_score,
                self.max_center_dist,
                prev_frame_id_inds,
                curr_frame_boxes,
                prev_frame_boxes,
            )

            match_i, match_j = linear_sum_assignment((-traj_score))

            track_ids = instance_ids.new_full((frame.num_detected,), -1)
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
                if n_traj >= self.max_tracks or traj_score[i, j] > thresh:
                    logger.debug(
                        f"Assigning instance {i} to track {j} with id {unique_ids[j]}"
                    )
                    track_ids[i] = unique_ids[j]
                    frame.instances[i].track_score = traj_score[i, j].item()
            logger.debug(f"track_ids: {track_ids}")
            for i in range(frame.num_detected):
                if track_ids[i] < 0:
                    logger.debug(f"Creating new track {curr_track}")
                    curr_track += 1
                    track_ids[i] = curr_track

            frame.matches = (match_i, match_j)

            for instance, track_id in zip(frame.instances, track_ids):
                instance.pred_track_id = track_id

            tracked_frames.append(frame)

            if frame.has_instances():
                self.track_queue.add_frame(frame)
                self.num_frames_tracked += 1
            else:
                self.track_queue.increment_gaps([])

        return tracked_frames
    

    def _run_frame_by_frame_tracker(
        self, model: GlobalTrackingTransformer, frames: list[Frame]
    ) -> Frame:
        """Run global tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames containing reid features. See `dreem.io.data_structures` for more info.

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

        query_ind = len(frames) - 1
        query_frame = frames[query_ind]

        query_instances = query_frame.instances
        all_instances = [instance for frame in frames for instance in frame.instances]

        logger.debug(f"Frame {query_frame.frame_id.item()}")

        instances_per_frame = [frame.num_detected for frame in frames]

        total_instances, window_size = sum(instances_per_frame), len(
            instances_per_frame
        )  # Number of instances in window; length of window.

        logger.debug(f"total_instances: {total_instances}")

        overlap_thresh = self.overlap_thresh
        mult_thresh = self.mult_thresh
        n_traj = self.track_queue.n_tracks
        curr_track = self.track_queue.curr_track

        reid_features = torch.cat([frame.get_features() for frame in frames], dim=0)[
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

        instance_ids = torch.cat(
            [
                x.get_pred_track_ids()
                for batch_idx, x in enumerate(frames)
                if batch_idx != query_ind
            ],
            dim=0,
        ).view(
            n_nonquery
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
        )
        nonquery_boxes_px = torch.cat(
            [
                instance.bbox
                for nonquery_frame in frames
                if nonquery_frame.frame_id != query_frame.frame_id
                for instance in nonquery_frame.instances
            ],
            dim=0,
        )

        pred_boxes = model_utils.get_boxes(all_instances)
        query_boxes = pred_boxes[query_inds]  # n_k x 4
        nonquery_boxes = pred_boxes[nonquery_inds]  # n_nonquery x 4

        unique_ids = torch.unique(instance_ids)  # (n_nonquery,)

        logger.debug(f"Instance IDs: {instance_ids}")
        logger.debug(f"unique ids: {unique_ids}")

        id_inds = (
            unique_ids[None, :] == instance_ids[:, None]
        ).float()  # (n_nonquery, n_traj)

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        traj_score = post_processing.weight_decay_time(
            asso_nonquery, self.decay_time, reid_features, window_size, query_ind
        )

        if self.decay_time is not None and self.decay_time > 0:
            decay_time_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=nonquery_inds
            )

            decay_time_traj_score.index.name = "Query Instances"
            decay_time_traj_score.columns.name = "Nonquery Instances"

            query_frame.add_traj_score("decay_time", decay_time_traj_score)
        ################################################################################

        # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_query x n_traj
        traj_score = torch.mm(traj_score, id_inds.cpu())  # (n_query, n_traj)

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

            last_inds = (
                id_inds * torch.arange(n_nonquery, device=id_inds.device)[:, None]
            ).max(dim=0)[
                1
            ]  # M

            last_boxes = nonquery_boxes[last_inds]  # n_traj x 4
            last_ious = post_processing._pairwise_iou(
                Boxes(query_boxes), Boxes(last_boxes)
            )  # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)

        traj_score = post_processing.weight_iou(traj_score, self.iou, last_ious.cpu())

        if self.iou is not None and self.iou != "":
            iou_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
            )

            iou_traj_score.index.name = "Current Frame Instances"
            iou_traj_score.columns.name = "Unique IDs"

            query_frame.add_traj_score("weight_iou", iou_traj_score)
        ################################################################################

        # threshold for continuing a tracking or starting a new track -> they use 1.0
        # todo -> should also work without pos_embed
        traj_score = post_processing.filter_max_center_dist(
            traj_score,
            self.max_center_dist,
            id_inds,
            query_boxes_px,
            nonquery_boxes_px,
        )

        if self.max_center_dist is not None and self.max_center_dist > 0:
            max_center_dist_traj_score = pd.DataFrame(
                traj_score.clone().numpy(), columns=unique_ids.cpu().numpy()
            )

            max_center_dist_traj_score.index.name = "Current Frame Instances"
            max_center_dist_traj_score.columns.name = "Unique IDs"

            query_frame.add_traj_score("max_center_dist", max_center_dist_traj_score)

        ################################################################################
        scaled_traj_score = torch.softmax(traj_score, dim=1)
        scaled_traj_score_df = pd.DataFrame(
            scaled_traj_score.numpy(), columns=unique_ids.cpu().numpy()
        )
        scaled_traj_score_df.index.name = "Current Frame Instances"
        scaled_traj_score_df.columns.name = "Unique IDs"

        query_frame.add_traj_score("scaled", scaled_traj_score_df)
        ################################################################################

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
            if n_traj >= self.max_tracks or traj_score[i, j] > thresh:
                logger.debug(
                    f"Assigning instance {i} to track {j} with id {unique_ids[j]}"
                )
                track_ids[i] = unique_ids[j]
                query_frame.instances[i].track_score = scaled_traj_score[i, j].item()
        logger.debug(f"track_ids: {track_ids}")
        for i in range(n_query):
            if track_ids[i] < 0:
                logger.debug(f"Creating new track {curr_track}")
                curr_track += 1
                track_ids[i] = curr_track

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
