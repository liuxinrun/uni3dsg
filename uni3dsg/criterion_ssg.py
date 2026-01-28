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
class SSGCriterion(nn.Module):
    def __init__(self, 
                 obj_loss_weight: Optional[float] = 0.1,
                 rel_loss_weight: Optional[float] = 1.0,
                 use_focal_loss: bool = True):
        super().__init__()
        
        self.obj_loss_weight = obj_loss_weight
        self.rel_loss_weight = rel_loss_weight
        

        w_node_cls = np.ones(20)

        w_node_cls_full = np.ones(len(w_node_cls) + 1, dtype=np.float32)
        w_node_cls_full[-1] = 0.1 
        

        w_edge_foreground = np.ones(8)

        w_edge_foreground = np.ones(len(w_edge_foreground), dtype=np.float32)
        w_edge_foreground[0] = 0.05 

        self.register_buffer('w_node_cls', torch.from_numpy(w_node_cls_full).float())
        self.register_buffer('w_edge_cls', torch.from_numpy(w_edge_foreground).float())

        # 定义 Loss
        self.loss_node_cls = nn.CrossEntropyLoss()
        
        if use_focal_loss:
            self.loss_rel_cls = FocalLoss(weight=self.w_edge_cls, gamma=2.0)
        else:
            self.loss_rel_cls = nn.CrossEntropyLoss(weight=self.w_edge_cls)

            
    def __call__(self, obj_preds, rel_preds, rel_labels, proposal_labels):
 
        if isinstance(obj_preds, list):
            obj_pred_cat = torch.cat(obj_preds, dim=0)
            proposal_label_cat = torch.cat(proposal_labels, dim=0)
        else:
            obj_pred_cat = obj_preds
            proposal_label_cat = proposal_labels

        if isinstance(rel_preds, list):
            rel_pred_cat = torch.cat(rel_preds, dim=0)
            rel_label_cat = torch.cat(rel_labels, dim=0)
        else:
            rel_pred_cat = rel_preds
            rel_label_cat = rel_labels
        loss_obj = self.loss_node_cls(obj_pred_cat, proposal_label_cat)

        loss_rel = self.loss_rel_cls(rel_pred_cat, rel_label_cat)
        loss_values = [self.obj_loss_weight * loss_obj, self.rel_loss_weight * loss_rel]
        loss_name = ['loss_obj', 'loss_rel']
        
        return dict(zip(loss_name, loss_values))