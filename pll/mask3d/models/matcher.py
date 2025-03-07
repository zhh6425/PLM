# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def batch_l1_loss(inputs: torch.Tensor, targets: torch.Tensor):
    l1_loss = torch.abs(inputs[:, None, :] - targets[None, :, :]).sum(dim=-1)
    return l1_loss

batch_l1_loss_jit = torch.jit.script(
    batch_l1_loss
)  # type: torch.jit.ScriptModule

def batch_giou_loss(inputs: torch.Tensor, targets: torch.Tensor):

    inputs = inputs[:, None, :]  # (N, 1, 6)
    targets = targets[None, :, :]  # (1, M, 6)

    x1, y1, z1, h1, w1, v1 = inputs.unbind(dim=-1)  # (N, 1)
    x2, y2, z2, h2, w2, v2 = targets.unbind(dim=-1)  # (1, M)

    inter_x = torch.clamp(torch.min(x1 + w1 / 2, x2 + w2 / 2) - torch.max(x1 - w1 / 2, x2 - w2 / 2), min=0)
    inter_y = torch.clamp(torch.min(y1 + h1 / 2, y2 + h2 / 2) - torch.max(y1 - h1 / 2, y2 - h2 / 2), min=0)
    inter_z = torch.clamp(torch.min(z1 + v1 / 2, z2 + v2 / 2) - torch.max(z1 - v1 / 2, z2 - v2 / 2), min=0)
    inter_vol = inter_x * inter_y * inter_z  # (N, M)

    vol1 = h1 * w1 * v1  # (N, 1)
    vol2 = h2 * w2 * v2  # (1, M)
    union_vol = vol1 + vol2 - inter_vol  # (N, M)

    iou = inter_vol / (union_vol + 1e-6)

    enclosing_x = torch.max(x1 + w1 / 2, x2 + w2 / 2) - torch.min(x1 - w1 / 2, x2 - w2 / 2)
    enclosing_y = torch.max(y1 + h1 / 2, y2 + h2 / 2) - torch.min(y1 - h1 / 2, y2 - h2 / 2)
    enclosing_z = torch.max(z1 + v1 / 2, z2 + v2 / 2) - torch.min(z1 - v1 / 2, z2 - v2 / 2)
    enclosing_vol = enclosing_x * enclosing_y * enclosing_z

    giou = iou - (enclosing_vol - union_vol) / (enclosing_vol + 1e-6)
    giou_loss = 1 - giou  # (N, M)

    return giou_loss

batch_giou_loss_jit = torch.jit.script(
    batch_giou_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        cost_l1: float = 1,
        cost_bbox: float = 1,
        matching_margin: float = 0.5,
        num_points: int = 0,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_l1 = cost_l1
        self.cost_bbox = cost_bbox
        self.matching_margin = matching_margin

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_type):
        """More memory-friendly matching"""
        # bs, num_queries = outputs["pred_logits"].shape[:2]
        bs = len(outputs["pred_logits"])

        indices = []

        # Iterate through batch size
        for b in range(bs):
            num_queries = outputs["pred_logits"][b].shape[0]

            if targets[b]["labels"].numel() == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            out_prob = outputs["pred_logits"][b].softmax(
                -1
            )  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"].clone()

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            filter_ignore = tgt_ids == 253
            tgt_ids[filter_ignore] = 0
            cost_class = 1 - out_prob[:, tgt_ids]  # to prevent negtive value
            cost_class[
                :, filter_ignore
            ] = (
                -1.0
            )  # for ignore classes pretend perfect match ;) TODO better worst class match?

            out_mask = outputs["pred_masks"][
                b
            ].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b][mask_type].to(out_mask)

            if self.num_points != -1:
                point_idx = torch.randperm(
                    tgt_mask.shape[1], device=tgt_mask.device
                )[: int(self.num_points * tgt_mask.shape[1])]
                # point_idx = torch.randint(0, tgt_mask.shape[1], size=(self.num_points,), device=tgt_mask.device)
            else:
                # sample all points
                point_idx = torch.arange(
                    tgt_mask.shape[1], device=tgt_mask.device
                )

            out_bbox = outputs["pred_boxes"][b]
            tgt_bbox = targets[b]["bbox"]

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )
                cost_l1 = batch_l1_loss_jit(
                    out_bbox, tgt_bbox[:, :6]
                )

                cost_bbox = batch_giou_loss_jit(
                    out_bbox, tgt_bbox[:, :6]
                )

            # Final cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
                + self.cost_l1 * cost_l1
                + self.cost_bbox * cost_bbox
            )
            C = C.reshape(num_queries, -1).cpu()

            # modifine Hungarian to accapt multiple match
            pred_indices, tgt_indices = linear_sum_assignment(C)
            pred_indices = np.array(pred_indices)
            tgt_indices = np.array(tgt_indices)

            baseline_cost = {j: C[i, j] for i, j in zip(pred_indices, tgt_indices)}
            gt_threshold = {
                j: (lambda best_cost: best_cost + self.matching_margin * abs(best_cost))(baseline_cost[j])
                for j in tgt_indices
            }

            matches = list(zip(pred_indices, tgt_indices))
            matched_queries = set(pred_indices.tolist())

            while True:
                remaining = [i for i in range(num_queries) if i not in matched_queries]
                if not remaining:
                    break
                C_sub = C[remaining, :]
                sub_pred, sub_tgt = linear_sum_assignment(C_sub)
                sub_pred, sub_tgt = np.array(sub_pred), np.array(sub_tgt)

                new_found = False
                for local_idx, j in zip(sub_pred, sub_tgt):
                    real_idx = remaining[local_idx]
                    if tgt_ids[j] != 0:  # only the gt object
                        continue
                    if C[real_idx, j] <= gt_threshold[j]:
                        matches.append((real_idx, j))
                        matched_queries.add(real_idx)
                        new_found = True
                if not new_found:
                    break

            pred_tensor = torch.as_tensor([m[0] for m in matches], dtype=torch.int64)
            tgt_tensor = torch.as_tensor([m[1] for m in matches], dtype=torch.int64)
            indices.append((pred_tensor, tgt_tensor))

        return indices

    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
