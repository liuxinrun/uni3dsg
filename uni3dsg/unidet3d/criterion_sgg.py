import torch
from typing import Optional, Dict, List
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import math
from .structures import InstanceData_
from mmcv.ops import diff_iou_rotated_3d
from mmdet3d.registry import MODELS, TASK_UTILS
import numpy as np
from .criterion import _bbox_to_loss
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight.to(inputs.device), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


@MODELS.register_module()
class E2ESGGCriterion(nn.Module):
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

    def __init__(self, matcher, rel_matcher, loss_weight, non_object_weight,
                 iter_matcher, bbox_loss_simple, bbox_loss_rotated,
                 datasets, datasets_weights, topk):
        super().__init__()
        self.bbox_loss_simple = MODELS.build(bbox_loss_simple)
        self.bbox_loss_rotated = MODELS.build(bbox_loss_rotated)
        self.matcher = TASK_UTILS.build(matcher)
        self.connectivity_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.rel_matcher = TASK_UTILS.build(rel_matcher)
        self.non_object_weight = non_object_weight
        self.loss_weight = loss_weight
        self.iter_matcher = iter_matcher
        self.datasets = datasets 
        self.datasets_weights = datasets_weights
        self.topk = topk
        self.rel_topk = 6
        w_node_cls = np.ones(20)
        w_node_cls_full = np.ones(len(w_node_cls) + 1, dtype=np.float32)
        w_node_cls_full[:-1] = w_node_cls
        w_node_cls_full[-1] = 0.1 
        w_edge_foreground = np.ones(8) # 假设40个前景关系类
        w_edge_foreground = np.ones(len(w_edge_foreground), dtype=np.float32)
        self.register_buffer('w_node_cls', torch.from_numpy(w_node_cls_full).float())
        self.register_buffer('w_edge_cls', torch.from_numpy(w_edge_foreground).float())
        self.loss_node_cls = nn.CrossEntropyLoss(weight=self.w_node_cls)
        self.loss_rel_cls = FocalLoss(weight=self.w_edge_cls, gamma=2.0)


    def get_layer_loss(self, aux_outputs, insts, 
                       datasets_names, indices=None, gt=False):
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
        sub_pred_bboxes = aux_outputs['sub_pred_bboxes']
        obj_pred_bboxes = aux_outputs['obj_pred_bboxes']
        sub_cls_preds = aux_outputs['sub_cls_preds']
        obj_cls_preds = aux_outputs['obj_cls_preds']
        rel_preds = aux_outputs['rel_preds']
        if indices is None:
            indices, rel_indices = [], []
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
                indice = self.matcher(pred_instances, gt_instances, 
                                            self.topk[idx])
                indices.append(indice)     
                if not gt:
                    rel_pred_instances = InstanceData_(
                        sub_bboxes=sub_pred_bboxes[i],
                        obj_bboxes=obj_pred_bboxes[i],
                        sub_scores=sub_cls_preds[i],
                        obj_scores=obj_cls_preds[i],
                        rel_scores=rel_preds[i])
                    rel_gt_instances = InstanceData_(
                        labels=insts[i].labels_3d,
                        relationships=insts[i].relationships,
                        query_masks=insts[i].query_masks,
                        bboxes=torch.cat((insts[i].bboxes_3d.gravity_center, 
                                        insts[i].bboxes_3d.tensor[:, 3:] if \
                                        insts[i].bboxes_3d.with_yaw else \
                                        insts[i].bboxes_3d.tensor[:, 3:6]),
                                        dim=1))
                    rel_indice = self.rel_matcher(rel_pred_instances, rel_gt_instances, 
                                                self.rel_topk)
                    rel_indices.append(rel_indice)   
                else:
                    if insts[i].relationships.shape[0]==0:
                        rel_indices.append((rel_preds[i].new_empty((0,)), rel_preds[i].new_empty((0,))))
                    else:
                        rel_indices.append([torch.arange(len(sub_pred_bboxes[i])), torch.arange(len(sub_pred_bboxes[i])) % len(insts[i].relationships)])          

        cls_losses, sub_cls_losses, obj_cls_losses, rel_losses = [], [], [], []
        bbox_losses, sub_bbox_losses, obj_bbox_losses = [], [], []
        physics_losses = []
        for dataset_name, cls_pred, sub_cls_pred, obj_cls_pred, rel_pred, bbox, sub_bbox, obj_bbox, inst, (idx_q, idx_gt), (rel_idx_q, rel_idx_gt) in \
                zip(datasets_names, cls_preds, sub_cls_preds, obj_cls_preds, rel_preds, pred_bboxes, sub_pred_bboxes, obj_pred_bboxes, insts, indices, rel_indices):
            num_classes = cls_pred.shape[1] - 1
            num_rel_classes = rel_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), num_classes, dtype=torch.long)
            rel_target = cls_pred.new_full(
                (len(rel_pred),), 0, dtype=torch.long)
            sub_cls_target = sub_cls_pred.new_full(
                (len(sub_cls_pred),), num_classes, dtype=torch.long)
            obj_cls_target = obj_cls_pred.new_full(
                (len(obj_cls_pred),), num_classes, dtype=torch.long)
            gt_rels = torch.tensor(inst.relationships, device=rel_pred.device)
           
            if len(inst) == 0 or len(idx_q) == 0:
                cls_target = cls_pred.new_full((len(cls_pred),), num_classes, dtype=torch.long)
            else:
                cls_target[idx_q] = inst.labels_3d[idx_gt]
            if len(inst) == 0 or len(rel_idx_q) == 0:
                sub_cls_target = sub_cls_pred.new_full((len(sub_cls_pred),), num_classes, dtype=torch.long)
                obj_cls_target = obj_cls_pred.new_full((len(obj_cls_pred),), num_classes, dtype=torch.long)
                rel_target = rel_pred.new_full((len(rel_pred),), 0, dtype=torch.long)
            else:
                sub_cls_target[rel_idx_q] = inst.labels_3d[gt_rels[:,0]][rel_idx_gt]
                obj_cls_target[rel_idx_q] = inst.labels_3d[gt_rels[:,1]][rel_idx_gt]
                rel_target[rel_idx_q] = gt_rels[rel_idx_gt, 2]
            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            cls_losses.append(weight * F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor([1] * num_classes + \
                                                          [self.non_object_weight])))
            sub_cls_losses.append(weight * self.loss_node_cls(sub_cls_pred, sub_cls_target))
            obj_cls_losses.append(weight * self.loss_node_cls(obj_cls_pred, obj_cls_target))
            rel_losses.append(weight * self.loss_rel_cls(rel_pred, rel_target))
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
            if len(rel_idx_q) == 0:
                continue
            pred_sub_bbox = sub_bbox[rel_idx_q]
            pred_obj_bbox = obj_bbox[rel_idx_q]
            tgt_bbox = inst.bboxes_3d
            tgt_bbox = torch.cat((tgt_bbox.gravity_center, 
                                  tgt_bbox.tensor[:, 3:] if \
                                  tgt_bbox.with_yaw else \
                                  tgt_bbox.tensor[:, 3:6]),
                                  dim=1)  
            gt_rels = torch.tensor(inst.relationships, device=rel_pred.device)
            sub_tgt_bbox = tgt_bbox[gt_rels[:,0]][rel_idx_gt]        
            obj_tgt_bbox = tgt_bbox[gt_rels[:,1]][rel_idx_gt]
            if tgt_bbox.shape[1] == 7: # rotated case
                sub_bbox_losses.append(weight * self.bbox_loss_rotated(
                                        _bbox_to_loss(pred_sub_bbox), 
                                        _bbox_to_loss(sub_tgt_bbox)).mean())
                obj_bbox_losses.append(weight * self.bbox_loss_rotated(
                                        _bbox_to_loss(pred_obj_bbox), 
                                        _bbox_to_loss(obj_tgt_bbox)).mean())
            else:
                sub_bbox_losses.append(weight * self.bbox_loss_simple(
                                        _bbox_to_loss(pred_sub_bbox), 
                                        _bbox_to_loss(sub_tgt_bbox)).mean())
                obj_bbox_losses.append(weight * self.bbox_loss_simple(
                                        _bbox_to_loss(pred_obj_bbox), 
                                        _bbox_to_loss(obj_tgt_bbox)).mean())
        cls_loss = torch.mean(torch.stack(cls_losses))
        sub_cls_loss = torch.mean(torch.stack(sub_cls_losses))
        obj_cls_loss = torch.mean(torch.stack(obj_cls_losses))
        rel_loss = torch.mean(torch.stack(rel_losses))
        if len(bbox_losses) > 0:
            bbox_loss = torch.stack(bbox_losses).mean()
        else:
            bbox_loss = torch.tensor(0)
        if len(sub_bbox_losses) > 0:
            sub_bbox_loss = torch.stack(sub_bbox_losses).mean()
            obj_bbox_loss = torch.stack(obj_bbox_losses).mean()
        else:
            sub_bbox_loss = torch.tensor(0)
            obj_bbox_loss = torch.tensor(0)

        loss_value = [self.loss_weight[0] * cls_loss, self.loss_weight[1] * bbox_loss, self.loss_weight[0] * sub_cls_loss, self.loss_weight[1] * sub_bbox_loss, self.loss_weight[0] * obj_cls_loss, self.loss_weight[1] * obj_bbox_loss, self.loss_weight[0] * rel_loss]
        loss = sum(loss_value)
        loss_name=['cls_loss', 'bbox_loss', 'sub_cls_loss', 'sub_bbox_loss', 'obj_cls_loss', 'obj_bbox_loss', 'rel_loss']
        loss_dict = dict(zip(loss_name, loss_value))

        return loss, loss_dict, indices, rel_indices
    
    def __call__(self, pred, insts, datasets_names, gt_branch=None):
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
        loss_name=['cls_loss', 'bbox_loss', 'sub_cls_loss', 'sub_bbox_loss', 'obj_cls_loss', 'obj_bbox_loss', 'rel_loss']
        loss, loss_dict, indices, rel_indices = self.get_layer_loss(pred, insts, datasets_names)
        if gt_branch is not None:
            loss_gt, loss_dict_gt, indices_gt, rel_indices_gt = self.get_layer_loss(gt_branch, insts, datasets_names, None, gt=True)
            loss += loss_gt
            for k in loss_name:
                loss_dict[k] = loss_dict_gt[k]
            if 'aux_outputs' in gt_branch:
                for aux_gt_outputs in gt_branch['aux_outputs']:
                    loss_gt_aux, loss_gt_dict_aux, indices_gt_aux, rel_indices_gt_aux = self.get_layer_loss(aux_gt_outputs, insts, 
                                                datasets_names, None, gt=True)
                    loss += loss_gt_aux
                    for k in loss_name:
                        loss_dict[k]+=loss_gt_dict_aux[k]

        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for aux_outputs in pred['aux_outputs']:
                loss_aux, loss_dict_aux, indices_aux, rel_indices_aux = self.get_layer_loss(aux_outputs, insts, 
                                            datasets_names, indices)
                loss += loss_aux
                for k in loss_name:
                    loss_dict[k]+=loss_dict_aux[k]
                # if "rel_preds" in aux_outputs:
                #     loss_rel_aux, loss_dict_rel_aux = self.get_relations_loss(aux_outputs, insts, datasets_names, indices_aux)
                #     loss += loss_rel_aux
                #     for k in loss_rel_name:
                #         loss_dict[k] = loss_dict_rel_aux[k]

        # loss_dict['total_loss'] = loss
        return loss_dict

@TASK_UTILS.register_module()
class E2ESGGMatcher:
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
        relationships = torch.tensor(gt_instances.relationships, device=pred_instances.rel_scores.device)
        n_gts = len(relationships)
        if n_gts == 0:
            return relationships.new_empty((0,)), relationships.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        cost_value = torch.stack(cost_values).sum(dim=0)
        obj_query_masks = gt_instances.query_masks 
  
        mask_sub = obj_query_masks[relationships[:, 0].long(), :]

        mask_obj = obj_query_masks[relationships[:, 1].long(), :]

        rel_query_masks = torch.logical_or(mask_sub, mask_obj)
        cost_value = torch.where(rel_query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return (ids[:, 0], ids[:, 1])
    
    
@TASK_UTILS.register_module()
class RelBboxCost:
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
        sub_pred_bboxes = pred_instances.sub_bboxes.unsqueeze(axis=1).repeat(1, len(gt_instances.relationships[:, 0]), 1)
        obj_pred_bboxes = pred_instances.obj_bboxes.unsqueeze(axis=1).repeat(1, len(gt_instances.relationships[:, 1]), 1)
        sub_gt_bboxes = gt_instances.bboxes[gt_instances.relationships[:, 0]].unsqueeze(axis=0).repeat(sub_pred_bboxes.shape[0], 1, 1)
        obj_gt_bboxes = gt_instances.bboxes[gt_instances.relationships[:, 1]].unsqueeze(axis=0).repeat(obj_pred_bboxes.shape[0], 1, 1)
        assert gt_instances.bboxes.shape[1] == pred_instances.sub_bboxes.shape[1]  
        if gt_instances.bboxes.shape[1] == 7: #rotated case
            sub_cost = self.loss_rotated(_bbox_to_loss(sub_pred_bboxes), 
                                    _bbox_to_loss(sub_gt_bboxes))
            obj_cost = self.loss_rotated(_bbox_to_loss(obj_pred_bboxes), 
                                    _bbox_to_loss(obj_gt_bboxes))
        else:
            sub_cost = self.loss_simple(_bbox_to_loss(sub_pred_bboxes),
                                    _bbox_to_loss(sub_gt_bboxes))
            obj_cost = self.loss_simple(_bbox_to_loss(obj_pred_bboxes), 
                                    _bbox_to_loss(obj_gt_bboxes))
        return (sub_cost + obj_cost) * self.weight
    
@TASK_UTILS.register_module()
class RelQueryClassificationCost:
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
        sub_scores = pred_instances.sub_scores.softmax(-1)
        obj_scores = pred_instances.obj_scores.softmax(-1)
        sub_cost = -sub_scores[:, gt_instances.labels[gt_instances.relationships[:, 0]]]
        obj_cost = -obj_scores[:, gt_instances.labels[gt_instances.relationships[:, 1]]]
        return (sub_cost + obj_cost) * self.weight
    
@TASK_UTILS.register_module()
class RelClassificationCost:
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
        rel_scores = pred_instances.rel_scores.softmax(-1)
        cost = -rel_scores[:, gt_instances.relationships[:, 2]]
        return cost * self.weight
