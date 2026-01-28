import torch
from typing import Optional
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
from torch import Tensor
import torch.nn.functional as F
import math
from .structures import InstanceData_
from mmcv.ops import diff_iou_rotated_3d
from mmdet3d.registry import MODELS, TASK_UTILS

def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))
    
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

@MODELS.register_module()
class UniDet3DCriterion():
    """Universal 3D detection criterion.

    Args:
        matcher (Callable): A class or function for matching queries with 
            ground truth (GT) instances.
        loss_weight (List[float]): A list of two weights corresponding to 
            classification loss and regression loss, respectively.
        iter_matcher (bool): Flag indicating whether to use a separate 
            matcher for each encoder layer.
        bbox_loss_simple (dict): Configuration for bounding box loss 
            w/o angle.
        bbox_loss_rotated (dict): Configuration for bounding box loss 
            with angle.
        datasets (List[str]): A list of dataset names for each scene 
            in batch.
        datasets_weights (List[float]): A list of loss weights corresponding 
            to each dataset in `datasets`.
        topk (List[int]): A list of integer indicating the number 
            of top predictions to consider for each gt bbox 
            when computing losses.
    """

    def __init__(self, matcher, loss_weight, non_object_weight,
                 iter_matcher, bbox_loss_simple, bbox_loss_rotated,
                 datasets, datasets_weights, topk):
        self.bbox_loss_simple = MODELS.build(bbox_loss_simple)
        self.bbox_loss_rotated = MODELS.build(bbox_loss_rotated)
        self.matcher = TASK_UTILS.build(matcher)
        self.non_object_weight = non_object_weight
        self.loss_weight = loss_weight
        self.iter_matcher = iter_matcher
        self.datasets = datasets 
        self.datasets_weights = datasets_weights
        self.topk = topk

    def get_layer_loss(self, aux_outputs, insts, 
                       datasets_names, indices=None):
        """Per-layer auxiliary loss.

        Args:
            aux_outputs (Dict): A dictionary containing auxiliary outputs, with
                the following keys:
                - 'cls_preds' (List[Tensor]): A list of tensors of shape 
                (n_queries, n_classes + 1) for each sample in the batch, 
                representing predicted class scores.
                - 'bboxes' (List[Tensor]): A list of tensors of shape
                (n_queries, 7) for each sample in the batch, representing 
                the predicted bounding box coordinates.
            
            insts (List[InstanceData_]): A list of ground truth instances for 
                each sample in the batch, where each instance contains:
                - labels_3d (Tensor): Shape (n_gts_i,), containing the labels 
                for the ground truth bounding boxes.
                - bboxes_3d (DepthInstance3DBoxes): ground truth bounding boxes.

            datasets_names (List[str]): A list of dataset names corresponding 
                to each sample in the batch.

            indices (Optional[List[Tuple[Tensor]]]): Indices for matching 
                predicted bboxes with ground truth bboxes. If None, 
                these will be computed internally.

        Returns:
            Tensor: loss value.

        """
        cls_preds = aux_outputs['cls_preds']
        pred_bboxes = aux_outputs['bboxes']
        if indices is None:
            indices = []
            for i in range(len(insts)):
                idx = self.datasets.index(datasets_names[i])
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    bboxes=pred_bboxes[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    query_masks=insts[i].query_masks,
                    bboxes=torch.cat((insts[i].bboxes_3d.gravity_center, 
                                      insts[i].bboxes_3d.tensor[:, 3:] if \
                                      insts[i].bboxes_3d.with_yaw else \
                                      insts[i].bboxes_3d.tensor[:, 3:6]),
                                      dim=1))
                indices.append(self.matcher(pred_instances, gt_instances, 
                                            self.topk[idx]))                     

        cls_losses = []
        for dataset_name, cls_pred, inst, (idx_q, idx_gt) in \
                zip(datasets_names, cls_preds, insts, indices):
            num_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), num_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            if len(inst) == 0 or len(idx_q) == 0:
                cls_target = cls_pred.new_full((len(cls_pred),), num_classes, dtype=torch.long)
            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            cls_losses.append(weight * F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor([1] * num_classes + \
                                                          [self.non_object_weight])))
        cls_loss = torch.mean(torch.stack(cls_losses))

        bbox_losses = []
        for dataset_name, bbox, inst, (idx_q, idx_gt) in zip(datasets_names,
                                                pred_bboxes, insts, indices):
            if len(inst) == 0 or len(idx_q) == 0:
                continue
            pred_bbox = bbox[idx_q]
            tgt_bbox = inst.bboxes_3d[idx_gt]
            tgt_bbox = torch.cat((tgt_bbox.gravity_center, 
                                  tgt_bbox.tensor[:, 3:] if \
                                  tgt_bbox.with_yaw else \
                                  tgt_bbox.tensor[:, 3:6]),
                                  dim=1)     

            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            if tgt_bbox.shape[1] == 7: # rotated case
                bbox_losses.append(weight * self.bbox_loss_rotated(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
            else:
                bbox_losses.append(weight * self.bbox_loss_simple(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
        if len(bbox_losses):
            bbox_loss = torch.stack(bbox_losses).mean()
        else:
            bbox_loss = torch.tensor(0)

        loss_value = [self.loss_weight[0] * cls_loss, self.loss_weight[1] * bbox_loss]
        loss = sum(loss_value)
        loss_name=['cls_loss', 'bbox_loss']
        loss_dict = dict(zip(loss_name, loss_value))

        return loss, loss_dict

    def __call__(self, pred, insts, datasets_names):
        """Loss main function.

            pred (Dict): A dictionary containing auxiliary outputs, with
                the following keys:
                - 'cls_preds' (List[Tensor]): A list of tensors of shape 
                (n_queries, n_classes + 1) for each sample in the batch, 
                representing predicted class scores.
                - 'bboxes' (List[Tensor]): A list of tensors of shape
                (n_queries, 7) for each sample in the batch, representing 
                the predicted bounding box coordinates.
            
            insts (List[InstanceData_]): A list of ground truth instances for 
                each sample in the batch, where each instance contains:
                - labels_3d (Tensor): Shape (n_gts_i,), containing the labels 
                for the ground truth bounding boxes.
                - bboxes_3d (DepthInstance3DBoxes): ground truth bounding boxes.

            datasets_names (List[str]): A list of dataset names corresponding 
                to each sample in the batch.
        
        Returns:
            Dict: with instance loss value.
        """
        loss_name=['cls_loss', 'bbox_loss']
        loss, loss_dict = self.get_layer_loss(pred, insts, datasets_names)
        
        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for aux_outputs in pred['aux_outputs']:
                loss_aux, loss_dict_aux = self.get_layer_loss(aux_outputs, insts, 
                                            datasets_names, indices)
                loss += loss_aux
                for k in loss_name:
                    loss_dict[k]+=loss_dict_aux[k]
        return loss_dict

def _bbox_to_loss(bbox):
    """Transform box to the axis-aligned or rotated iou loss format.

    Args:
        bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

    Returns:
        Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
    """
    # # rotated iou loss accepts (x, y, z, w, h, l, heading)
    if bbox.shape[-1] != 6:
        return bbox

    # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)

@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gt_bboxes,).

        Returns:
            Tensor: Cost of shape (n_pred_bboxes, n_gt_bboxes).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight


@TASK_UTILS.register_module()
class QueryStableClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight, cec_beta, alpha, gamma):
        self.weight = weight
        self.cec_beta = cec_beta 
        self.alpha = alpha
        self.gamma = gamma   
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gt_bboxes,).

        Returns:
            Tensor: Cost of shape (n_pred_bboxes, n_gt_bboxes).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        assert gt_instances.bboxes.shape[1] == pred_instances.bboxes.shape[1]  
        if gt_instances.bboxes.shape[1] == 7: #rotated case
            iou = diff_iou_rotated_3d(_bbox_to_loss(pred_bboxes).unsqueeze(0), 
                                    _bbox_to_loss(gt_bboxes).unsqueeze(0))[0]
        else:
            iou = (AxisAlignedBboxOverlaps3D()(_bbox_to_loss(pred_bboxes), _bbox_to_loss(gt_bboxes), mode='giou') + 1) /2
        scores = pred_instances.scores.sigmoid()
        if self.cec_beta > 0:
            _s = scores
            _u = torch.zeros_like(_s)
            _u[:, gt_instances.labels] = iou
            # scale max(giou) to 1
            scalar = (1 / (_u.max() + 1e-8))
            scalar = torch.max(scalar, torch.ones_like(scalar))
            _u = _u * scalar
            # 
            scores = (_s * _u.pow(self.cec_beta))
            # cost = -scores[:, gt_instances.labels]

        neg_cost_class = (1 - self.alpha) * (scores**self.gamma) * (-(1 - scores + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - scores) ** self.gamma) * (-(scores + 1e-8).log())
        cost = pos_cost_class[:, gt_instances.labels] - neg_cost_class[:, gt_instances.labels]

        return cost * self.weight

@TASK_UTILS.register_module()
class BboxCostJointTraining:
    """Regression cost for bounding boxes.

    Args:
        weigth (float): Weight of the cost.
        bbox_loss_simple (dict): Configuration for 
            bounding box loss w/o angle.
        bbox_loss_rotated (dict): Configuration for 
            bounding box loss with angle.
    """
    def __init__(self, weight, loss_simple, loss_rotated):
        self.weight = weight
        self.loss_simple = MODELS.build(loss_simple)
        self.loss_rotated = MODELS.build(loss_rotated)

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `bboxes` of shape (n_pred_bboxes, 6) or 
                (n_pred_bboxes, 7).
            gt_instances (:obj:`InstanceData_`): Ground truth which must 
                contain `bboxes` of shape (n_gt_bboxes, 6) or 
                (n_gt_bboxes, 7).
        
        Returns:
            Tensor: Cost of shape (n_pred_bboxes, n_gt_bboxes).
        """
        pred_bboxes = pred_instances.bboxes.unsqueeze(axis=1).repeat(1, gt_instances.bboxes.shape[0], 1)
        gt_bboxes = gt_instances.bboxes.unsqueeze(axis=0).repeat(pred_bboxes.shape[0], 1, 1)
        assert gt_instances.bboxes.shape[1] == pred_instances.bboxes.shape[1]  
        if gt_instances.bboxes.shape[1] == 7: #rotated case
            cost = self.loss_rotated(_bbox_to_loss(pred_bboxes), 
                                    _bbox_to_loss(gt_bboxes))
        else:
            cost = self.loss_simple(_bbox_to_loss(pred_bboxes), 
                                    _bbox_to_loss(gt_bboxes))
        return cost * self.weight

@TASK_UTILS.register_module()
class UniMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
    """

    def __init__(self, costs):
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, topk, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `bboxes` of shape (n_pred_bboxes, 6), `scores`
                of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gt_bboxes,), `bboxes` of shape (n_gt_bboxes, 6),
            topk (int): Limit topk matches per bbox.

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]

@TASK_UTILS.register_module()
class SplitMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
    """

    def __init__(self, costs):
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, topk, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `bboxes` of shape (n_pred_bboxes, 6), `scores`
                of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gt_bboxes,), `bboxes` of shape (n_gt_bboxes, 6),
            topk (int): Limit topk matches per bbox.

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,)), labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        seg_values = torch.topk(
            cost_value, 1 + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        det_values = torch.topk(
            cost_value, topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        seg_ids = torch.argwhere(cost_value < seg_values)
        det_ids = torch.argwhere(cost_value < det_values)
        return seg_ids[:, 0], seg_ids[:, 1], det_ids[:, 0], det_ids[:, 1]

@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]