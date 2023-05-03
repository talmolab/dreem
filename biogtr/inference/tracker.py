import torch
import pandas as pd
from typing import Union
from torch.nn import Module
from biogtr.inference import post_processing
from biogtr.inference.boxes import Boxes
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from scipy.optimize import linear_sum_assignment


class Tracker(Module):
    def __init__(
        self,
        model: GlobalTrackingTransformer,
        use_vis_feats: bool,
        overlap_thresh: float = 0.01,
        mult_thresh: bool = True,
        decay_time: float = None,
        iou: str = None,
        max_center_dist: float = None,
    ):
        """
        Initialize a tracker to run inference
        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            use_vis_feats: Whether or not to use visual feature extractor
            overlap_thresh: the trajectory overlap threshold to be used for assignment
            mult_thresh: Whether or not to use weight threshold
            decay_time: weight for `decay_time` postprocessing
            iou: Either [None, '', "mult" or "max"] Whether to use multiplicative or max iou reweighting
            max_center_dist: distance threshold for filtering trajectory score matrix
        """
        self.model = model
        (_,) = self.model.eval()
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist

    def forward(self, instances: dict, all_instances: list = None):
        """
        Run tracker and get predicted trajectories
        Returns: instances dict populated with pred track ids and association matrix scores
        Args:
            instances: data dict to run inference on
            all_instances: list of instances from previous chunks to stitch together full trajectory
        """
        # Extract feature representations with pre-trained encoder.
        for frame in instances:
            if (frame["num_detected"] > 0).item():
                if not self.use_vis_feats:
                    num_frame_instances = frame["crops"].shape[0]
                    frame["features"] = torch.zeros(
                        num_frame_instances, self.model.d_model
                    )
                    # frame["features"] = torch.randn(
                    #     num_frame_instances, self.model.d_model
                    # )

                # comment out to turn encoder off

                # Assuming the encoder is already trained or train encoder jointly.
                else:
                    with torch.no_grad():
                        z = self.model.encoder_model(frame["crops"])
                        frame["features"] = z
        """
        I feel like this chunk is unnecessary:
        reid_features = torch.cat(
            [frame["features"] for frame in instances], dim=0
        ).unsqueeze(0)

        asso_preds, pred_boxes, pred_time, embeddings = self.model(
            instances, reid_features
        )
        """
        return self.sliding_inference(
            instances, window_size=self.window_size, all_instances=all_instances
        )

    def sliding_inference(self, instances, window_size, all_instances=None):
        """Performs sliding inference on the input video (instances) with a given
        window size.
        Args:
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
            window_size: An integer.
        Returns:
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
        # ------------------------- An example of instances ------------------------ #
        D: embedding dimension.
        N_i: number of detected instances in i-th frame of window.
        instances = [
            {
                # Each dictionary is a frame.
                "frame_id": frame index int,
                "num_detected": N_i,
                "gt_track_ids": (N_i,),
                "poses": (N_i, 13, 2),  # 13 keypoints for the pose (x, y) coords.
                "bboxes": (N_i, 4),  # in pascal_voc unrounded unnormalized
                "features": (N_i, D),  # Features are deleted but can optionally be kept if need be.
                "pred_track_ids": (N_i,),  # Filled out after sliding_inference.
            },
            {},  # Frame 2.
            ...
        ]
        """

        # B: batch size.
        # D: embedding dimension.
        # nc: number of channels.
        # H: height.
        # W: width.

        video_len = len(instances)
        id_count = 0

        for frame_id in range(video_len):
            if frame_id == 0:
                if all_instances is not None and len(all_instances) != 0:
                    instances[0]["pred_track_ids"] = torch.arange(
                        0, len(all_instances[-1]["bboxes"])
                    )
                    id_count = len(all_instances[-1]["bboxes"])

                    test = [all_instances[-1], instances[0]]

                    test, id_count = self._run_global_tracker(
                        test,
                        k=1,
                        id_count=id_count,
                        overlap_thresh=self.overlap_thresh,
                        mult_thresh=self.mult_thresh,
                    )

                    instances[0] = test[-1]

                    # print('first frame of new chunk!', instances[frame_id]['pred_track_ids'])
                else:
                    instances[0]["pred_track_ids"] = torch.arange(
                        0, len(instances[0]["bboxes"])
                    )
                    id_count = len(instances[0]["bboxes"])

                    # print('id count: ', id_count)
                    # print('first overall frame!', instances[frame_id]['pred_track_ids'])
            else:
                win_st = max(0, frame_id + 1 - window_size)
                win_ed = frame_id + 1
                instances[win_st:win_ed], id_count = self._run_global_tracker(
                    instances[win_st:win_ed],
                    k=min(window_size - 1, frame_id),
                    id_count=id_count,
                    overlap_thresh=self.overlap_thresh,
                    mult_thresh=self.mult_thresh,
                )
                # print(f'frame: {frame_id}', instances[frame_id]['pred_track_ids'])

            """
            # If first frame.
            if frame_id == 0:
                instances[0]["pred_track_ids"] = torch.arange(
                    0, len(instances[0]["bboxes"]))
                id_count = len(instances[0]["bboxes"])
            else:
                win_st = max(0, frame_id + 1 - window_size)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self._run_global_tracker(
                    instances[win_st: win_ed],
                    k=min(window_size - 1, frame_id),
                    id_count=id_count,
                    overlap_thresh=self.overlap_thresh,
                    mult_thresh=self.mult_thresh)
            """

            # If features are out of window, set to none.
            # if frame_id - window_size >= 0:
            #     instances[frame_id - window_size]["features"] = None

        # TODO: Insert postprocessing.

        # Remove last few features from cuda.
        for frame in instances[-window_size:]:
            frame["features"] = frame["features"].cpu()

        return instances

    def _run_global_tracker(self, instances, k, id_count, overlap_thresh, mult_thresh):
        """run_global_tracker performs the actual tracking (Hungarian algorithm) and
        track assigning.
        Args:
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
            k: An integer for the query frame within the window of instances.
            id_count: The count of total identities so far.
            overlap_thresh: A float number between 0 and 1 specifying how much
            overlap is necessary for assigning a new instance to an existing identity.
            mult_thresh: A boolean for whether or not multiple thresholds should be used.
            This is not functional as of now.
        Returns:
            instances: The exact list of dictionaries as before but with assigned track ids
            and new track ids for the query frame. Refer to the example for the structure.
            id_count: An integer for the updated identity count so far.
        # ------------------------- An example of instances ------------------------ #
        NOTE: This instances variable is the window subset of the instances variable in sliding_inference.
        *: each item in instances is a frame in the window. So it follows
            that each frame in the window has * detected instances.
        D: embedding dimension.
        N_i: number of detected instances in i-th frame of window.
        T: length of window.
        The features in instances can be of shape (2 to T, *, D) when stacked together.
        instances = [
            {
                # Each dictionary is a frame.
                "frame_id": frame index int,
                "num_detected": N_i,
                "gt_track_ids": (N_i,),
                "poses": (N_i, 13, 2),  # 13 keypoints for the pose (x, y) coords.
                "bboxes": (N_i, 4),  # in pascal_voc unrounded unnormalized
                "features": (N_i, D),
                "pred_track_ids": (N_i,),  # Before assignnment, these are all -1.
            },
            ...
        ]
        """

        # *: each item in instances is a frame in the window. So it follows
        #    that each frame in the window has * detected instances.
        # D: embedding dimension.
        # N: number of instances in the window.
        # N_i: number of detected instances in i-th frame of window.
        # n_t: a list of number of instances in each frame of the window.
        # N_t: number of instances in current/query frame (rightmost frame of the window).
        # Np: number of instances in the window excluding the current/query frame.
        # T: length of window.
        # L: number of decoder blocks.
        # M: number of existing tracks within the window so far.

        # Number of instances in each frame of the window.
        # E.g.: n_t: [4, 5, 6, 7]; window of length 4 with 4 detected instances in the first frame of the window.
        n_t = [frame["num_detected"] for frame in instances]

        N, T = sum(n_t), len(n_t)  # Number of instances in window; length of window.

        reid_features = torch.cat(
            [frame["features"].to(self.device) for frame in instances], dim=0
        )[
            None
        ]  # (1, N, D=512)

        asso_output, pred_boxes, pred_time, embeddings = self.transformer(
            instances, reid_features, k
        )  # (L=1, N_t, N)

        asso_output = asso_output[-1].split(n_t, dim=1)  # (T, N_t, N_i)
        asso_output = post_processing.softmax_asso(asso_output)  # (T, N_t, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (N_t, N)
        instances[k]["asso_output"] = asso_output

        N_t = instances[k][
            "num_detected"
        ]  # Number of instances in the current/query frame.

        N_p = (
            N - N_t
        )  # Number of instances in the window not including the current/query frame.

        ids = torch.cat(
            [x["pred_track_ids"] for t, x in enumerate(instances) if t != k], dim=0
        ).view(
            N_p
        )  # (N_p,)

        k_inds = [x for x in range(sum(n_t[:k]), sum(n_t[: k + 1]))]
        nonk_inds = [i for i in range(N) if i not in k_inds]

        asso_output = asso_output[:, nonk_inds]  # (N_t, N_p)

        k_boxes = pred_boxes[k_inds]  # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds]  # Np x 4

        # TODO: Insert postprocessing.

        unique_ids = torch.unique(ids)  # (M,)
        # M = len(unique_ids)  # Number of existing tracks.
        id_inds = (unique_ids[None, :] == ids[:, None]).float()  # (N_p, M)

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        # (n_k x Np) x (Np x M) --> n_k x M
        asso_output = post_processing.weight_decay_time(
            asso_output, self.decay_time, reid_features, T, k
        )

        asso_output = torch.mm(asso_output, id_inds)  # (N_t, M)

        instances[k]["decay_time_traj_score"] = pd.DataFrame(
            (asso_output).numpy(), columns=unique_ids.numpy()
        )
        instances[k]["decay_time_traj_score"].index.name = "Current Frame Instances"
        instances[k]["decay_time_traj_score"].columns.name = "Unique IDs"

        ################################################################################

        # with iou -> combining with location in tracker, they set to True
        # todo -> should also work without pos_embed

        if id_inds.numel() > 0:
            # this throws error, think we need to slice?
            # last_inds = (id_inds * torch.arange(
            #    N_p, device=id_inds.device)[:, None]).max(dim=0)[1] # M

            last_inds = (
                id_inds * torch.arange(N_p[0], device=id_inds.device)[:, None]
            ).max(dim=0)[
                1
            ]  # M

            last_boxes = nonk_boxes[last_inds]  # M x 4
            last_ious = self._pairwise_iou(Boxes(k_boxes), Boxes(last_boxes))  # n_k x M
        else:
            last_ious = asso_output.new_zeros(asso_output.shape)

        asso_output = post_processing.weight_iou(asso_output, self.iou, last_ious)

        instances[k]["with_iou_traj_score"] = pd.DataFrame(
            (asso_output).numpy(), columns=unique_ids.numpy()
        )
        instances[k]["with_iou_traj_score"].index.name = "Current Frame Instances"
        instances[k]["with_iou_traj_score"].columns.name = "Unique IDs"

        ################################################################################

        # threshold for continuing a tracking or starting a new track -> they use 1.0
        # todo -> should also work without pos_embed

        asso_output = post_processing.filter_max_center_dist(
            asso_output, self.max_center_dist, k_boxes, nonk_boxes, id_inds
        )

        instances[k]["max_center_dist_traj_score"] = pd.DataFrame(
            (asso_output).numpy(), columns=unique_ids.numpy()
        )
        instances[k][
            "max_center_dist_traj_score"
        ].index.name = "Current Frame Instances"
        instances[k]["max_center_dist_traj_score"].columns.name = "Unique IDs"

        ################################################################################

        match_i, match_j = linear_sum_assignment((-asso_output))

        track_ids = ids.new_full((N_t,), -1)

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
            if asso_output[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(N_t):
            if track_ids[i] < 0:
                track_ids[i] = id_count
                id_count += 1

        instances[k]["matches"] = (match_i, match_j)
        instances[k]["pred_track_ids"] = track_ids

        instances[k]["final_traj_score"] = pd.DataFrame(
            (asso_output).numpy(), columns=unique_ids.numpy()
        )
        instances[k]["final_traj_score"].index.name = "Current Frame Instances"
        instances[k]["final_traj_score"].columns.name = "Unique IDs"

        instances[k]["pos_emb"] = embeddings["pos"]
        instances[k]["temp_emb"] = embeddings["temp"]

        return instances, id_count

    def _pairwise_intersection(self, boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
        """
        Given two lists of boxes of size N and M,
        compute the intersection area between __all__ N x M pairs of boxes.
        The box order must be (xmin, ymin, xmax, ymax)
        Args:
            boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
        Returns:
            Tensor: intersection, sized [N,M].
        """
        boxes1, boxes2 = boxes1.tensor, boxes2.tensor
        width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
            boxes1[:, None, :2], boxes2[:, :2]
        )  # [N,M,2]

        width_height.clamp_(min=0)  # [N,M,2]
        intersection = width_height.prod(dim=2)  # [N,M]

        return intersection

    def _pairwise_iou(self, boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
        """
        Given two lists of boxes of size N and M, compute the IoU
        (intersection over union) between **all** N x M pairs of boxes.
        The box order must be (xmin, ymin, xmax, ymax).
        Args:
            boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
        Returns:
            Tensor: IoU, sized [N,M].
        """
        area1 = boxes1.area()  # [N]
        area2 = boxes2.area()  # [M]
        inter = self._pairwise_intersection(boxes1, boxes2)

        # handle empty boxes
        iou = torch.where(
            inter > 0,
            inter / (area1[:, None] + area2 - inter),
            torch.zeros(1, dtype=inter.dtype, device=inter.device),
        )
        return iou