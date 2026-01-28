# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
import math
from terminaltables import AsciiTable
from functools import reduce
from collections import defaultdict
from .criterion import _bbox_to_loss
from mmdet3d.structures import AxisAlignedBboxOverlaps3D

# ===========================================================================
# Helper: Unify Data Types to CPU Tensor
# ===========================================================================
def to_tensor(data):
    """Ensure data is a CPU tensor."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, list):
        return torch.tensor(data)
    return data

def bbox_overlaps_3d(boxes1, boxes2):
    """
    Calculate 3D IoU between two sets of Axis Aligned boxes.
    """
    if isinstance(boxes1, torch.Tensor): boxes1 = boxes1.cpu()
    if isinstance(boxes2, torch.Tensor): boxes2 = boxes2.cpu()
    iou_calculator = AxisAlignedBboxOverlaps3D()
    return iou_calculator(_bbox_to_loss(boxes1[:, :6]).float(), 
                          _bbox_to_loss(boxes2[:, :6]).float())

def get_rank(scores, gt_label):
    """
    Find the rank of the gt_label in the scores (descending order).
    scores: [Num_Classes]
    gt_label: int or Tensor scalar
    Returns: rank (1-based), or > len(scores) if not found
    """
    # Ensure inputs are on the same device and type
    if isinstance(gt_label, torch.Tensor):
        gt_label = gt_label.item()
    
    # Sort descending
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    # Find index
    # We use (sorted_indices == gt_label).nonzero() to find where the label landed
    matches = (sorted_indices == gt_label).nonzero(as_tuple=True)[0]
    
    if len(matches) == 0:
        return len(scores) + 1 # Not found (should not happen if label exists in classes)
    return matches.item() + 1

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def bbox_overlaps(boxes1, boxes2):
    """
    Calculate IoU between two sets of bounding boxes.
    boxes1: (N, 4) or (N, 7)
    boxes2: (M, 4) or (M, 7)
    Note: For 3D boxes (7-dim), we typically use BEV IoU or 3D IoU. 
    Here we simplify to BEV IoU (x, y, w, l) or AABB IoU for matching speed, 
    matching standard 2D evaluation logic.
    Assumes boxes are (x, y, z, w, l, h, rot) or (x1, y1, x2, y2).
    Let's assume AABB (Axis Aligned) for simplicity or input pre-processed boxes.
    """
    # Simplified AABB IoU for matching logic integration
    # You might need to replace this with mmdet3d's specific 3D IoU function if needed.
    # For now, assuming inputs are converted to (x1, y1, x2, y2) AABB for matching.
    
    # Check if inputs are tensors
    if isinstance(boxes1, torch.Tensor): boxes1 = boxes1.numpy()
    if isinstance(boxes2, torch.Tensor): boxes2 = boxes2.numpy()
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # If 3D boxes (7 dims: x, y, z, dx, dy, dz, rot), convert to BEV AABB (x1, y1, x2, y2)
    if boxes1.shape[1] >= 4: 
        # Placeholder: Assume input is already compatible or use a library function
        # Implementation of standard 2D IoU
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])
        wh = (rb - lt).clip(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        iou = inter / (area1[:, None] + area2 - inter)
        return iou
    return np.zeros((N, M))

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack(
            (
                class_scores[sub_id],
                predicate_scores,
                class_scores[ob_id],
            )
        )

    return triplets, triplet_boxes, triplet_scores

def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # 1. Class Match: Check if (Sub_Label, Pred_Label, Obj_Label) matches
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    
    # 2. Box Match: Check if IoU > threshold for both Subject and Object
    for gt_ind, gt_box, keep_inds in zip(
            np.where(gt_has_match)[0],
            gt_boxes[gt_has_match],
            keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        # gt_box: [x1_s, y1_s, x2_s, y2_s, x1_o, y1_o, x2_o, y2_o]
        sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
        obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

        inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

def _triplet_3d(relations, classes, boxes):
    """
    From (sub_idx, obj_idx, pred) -> (sub_label, pred_label, obj_label) triplets
    """
    if isinstance(classes, torch.Tensor):
        classes = classes.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(relations, torch.Tensor):
        relations = relations.cpu().numpy()      
    # --- 修改点：强制转换为 int，防止浮点数索引报错 ---
    relations = relations.astype(np.int64)     
    sub_id, obj_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2] 
    # 这里 classes[sub_id] 需要 sub_id 是整数
    triplets = np.column_stack((classes[sub_id], pred_label, classes[obj_id]))
    sub_boxes = boxes[sub_id]
    obj_boxes = boxes[obj_id]
    return triplets, sub_boxes, obj_boxes

def _triplet_tensor(relations, classes, boxes):
    """
    Construct triplets wholly on Tensor.
    relations: [K, 3] (sub_idx, obj_idx, pred_label)
    classes: [N]
    boxes: [N, 7]
    Returns:
        triplets: [K, 3] (sub_lbl, pred_lbl, obj_lbl)
        triplet_boxes: [K, 14] (sub_box, obj_box)
    """
    sub_id, obj_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    
    # Stack labels: [sub_lbl, pred_lbl, obj_lbl]
    triplets = torch.stack([classes[sub_id], pred_label, classes[obj_id]], dim=1)
    
    # Stack boxes: [sub_box(7), obj_box(7)] -> [K, 14]
    triplet_boxes = torch.cat([boxes[sub_id], boxes[obj_id]], dim=1)
    
    return triplets, triplet_boxes

def sub_obj_triplet_tensor(relations, sub_classes, sub_boxes, obj_classes, obj_boxes):
    """
    Construct triplets wholly on Tensor.
    relations: [K, 3] (sub_idx, obj_idx, pred_label)
    classes: [N]
    boxes: [N, 7]
    Returns:
        triplets: [K, 3] (sub_lbl, pred_lbl, obj_lbl)
        triplet_boxes: [K, 14] (sub_box, obj_box)
    """
    # sub_id, obj_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    
    # Stack labels: [sub_lbl, pred_lbl, obj_lbl]
    triplets = torch.stack([sub_classes, relations, obj_classes], dim=1)
    
    # Stack boxes: [sub_box(7), obj_box(7)] -> [K, 14]
    triplet_boxes = torch.cat([sub_boxes, obj_boxes], dim=1)

    return triplets, triplet_boxes

def intersect_2d_tensor(x1, x2):
    """
    Given two tensors [N, D] and [M, D], returns a [N, M] boolean tensor.
    True if rows match.
    Optimized for GPU.
    """
    # x1: [N, 1, D], x2: [1, M, D]
    # result: [N, M]
    if x1.numel() == 0 or x2.numel() == 0:
        return torch.zeros((x1.shape[0], x2.shape[0]), dtype=torch.bool, device=x1.device)
    return (x1.unsqueeze(1) == x2.unsqueeze(0)).all(dim=-1)

def _compute_pred_matches_3d_tensor(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres):
    """
    Compute matches entirely on Tensor.
    """
    # 1. Class Match Matrix [N_gt, N_pred]
    class_match = intersect_2d_tensor(pred_triplets, gt_triplets)
    
    # Early exit if no class matches
    if not class_match.any():
        return [[] for _ in range(pred_triplets.shape[0])]

    # 2. Box IoU Match
    # gt_boxes: [N_gt, 14], pred_boxes: [N_pred, 14]
    # Split into Subject and Object
    gt_s, gt_o = gt_boxes[:, :7], gt_boxes[:, 7:]
    pred_s, pred_o = pred_boxes[:, :7], pred_boxes[:, 7:]
    
    # Calculate IoU matrices [N_pred, N_gt]
    sub_iou = AxisAlignedBboxOverlaps3D()(_bbox_to_loss(pred_s[:,:6]).float(),
                                              _bbox_to_loss(gt_s[:,:6]).float())

    obj_iou = AxisAlignedBboxOverlaps3D()(_bbox_to_loss(pred_o[:,:6]).float(),
                                              _bbox_to_loss(gt_o[:,:6]).float())
    
    # 3. Combine: Class Match AND Sub_IoU > Thres AND Obj_IoU > Thres
    is_match = class_match & (sub_iou >= iou_thres) & (obj_iou >= iou_thres)
    
    # 4. Convert to List of List format for metric calculation
    # pred_to_gt[i] = list of GT indices that match prediction i
    pred_to_gt = [[] for _ in range(pred_triplets.shape[0])]
    
    # Find non-zero indices: (gt_idx, pred_idx)
    if is_match.any():
        pred_indices, gt_indices = torch.nonzero(is_match, as_tuple=True)
        
        # Convert to CPU list for iteration in the final metric reduction
        gt_indices = gt_indices.tolist()
        pred_indices = pred_indices.tolist()
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            pred_to_gt[pred_idx].append(gt_idx)
            
    return pred_to_gt

def _compute_pred_matches_3d(gt_triplets, pred_triplets, gt_sub_boxes, gt_obj_boxes,
    pred_sub_boxes, pred_obj_boxes, iou_thres=0.25):
    """Compute matches between GT and predicted 3D triplets"""
    triplet_match = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = triplet_match.any(1)
    pred_to_gt = [[] for _ in range(pred_triplets.shape[0])]
    for gt_ind, keep_row in enumerate(triplet_match):
        if not gt_has_match[gt_ind]:
            continue
        keep_inds = np.nonzero(keep_row)[0]
        if len(keep_inds) == 0:
            continue
        sub_iou = AxisAlignedBboxOverlaps3D()(_bbox_to_loss(torch.from_numpy(pred_sub_boxes[keep_inds][:,:6])).float(),
                                              _bbox_to_loss(torch.from_numpy(gt_sub_boxes[[gt_ind]][:,:6])).float()).numpy()

        obj_iou = AxisAlignedBboxOverlaps3D()(_bbox_to_loss(torch.from_numpy(pred_obj_boxes[keep_inds][:,:6])).float(),
                                              _bbox_to_loss(torch.from_numpy(gt_obj_boxes[[gt_ind]][:,:6])).float()).numpy()
        hit = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)
        for pi in keep_inds[hit.squeeze(-1)]:
            pred_to_gt[pi].append(int(gt_ind))
    return pred_to_gt

# ===========================================================================
# Part 1: Standard Detection Evaluation (Unchanged)
# ===========================================================================

def average_precision(recalls, precisions, mode='area'):
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap

def eval_det_cls(pred, gt, iou_thr=None):
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    image_ids = []
    confidence = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))
    confidence = np.array(confidence)
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]
        if len(BBGT) > 0:
            for j in range(len(BBGT)):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.
    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))
    return ret

def eval_map_recall(pred, gt, ovthresh=None):
    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]
    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[label][iou_idx]
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)
    return recall, precision, ap

def indoor_eval(gt_annos, dt_annos, metric, label2cat, logger=None, box_mode_3d=None):
    assert len(dt_annos) == len(gt_annos)
    pred = {}
    gt = {}
    for img_id in range(len(dt_annos)):
        det_anno = dt_annos[img_id]
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['bboxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred: pred[int(label)] = {}
            if img_id not in pred[label]: pred[int(label)][img_id] = []
            if label not in gt: gt[int(label)] = {}
            if img_id not in gt[label]: gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))
        gt_anno = gt_annos[img_id]
        gt_boxes = gt_anno['gt_bboxes_3d']
        labels_3d = gt_anno['gt_labels_3d']
        for i in range(len(labels_3d)):
            label = labels_3d[i]
            bbox = gt_boxes[i]
            if label not in gt: gt[label] = {}
            if img_id not in gt[label]: gt[label][img_id] = []
            gt[label][img_id].append(bbox)
    rec, prec, ap = eval_map_recall(pred, gt, metric)
    ret_dict = dict()
    header = ['classes']
    table_columns = [[label2cat[label] for label in ap[0].keys()] + ['Overall']]
    from mmengine.logging import print_log
    for i, iou_thresh in enumerate(metric):
        header.append(f'AP_{iou_thresh:.2f}')
        header.append(f'AR_{iou_thresh:.2f}')
        rec_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(ap[i][label][0])
        ret_dict[f'mAP_{iou_thresh:.2f}'] = float(np.nanmean(list(ap[i].values())))
        table_columns.append(list(map(float, list(ap[i].values()))))
        table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(rec[i][label][-1])
            rec_list.append(rec[i][label][-1])
        ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.nanmean(rec_list))
        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    return ret_dict

# ===========================================================================
# Part 2: Comprehensive Scene Graph Evaluation (Corrected)
# ===========================================================================

def get_metrics_all(confusion: np.array, VALID_CLASS_IDS: list, CLASS_LABELS: list):
    outputs = dict()
    ious = dict()
    precisions = dict()
    recalls = dict()
    if len(VALID_CLASS_IDS) == 0:
        VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        ious[label_name], precisions[label_name], recalls[label_name] \
            = get_metrics(label_id, confusion, VALID_CLASS_IDS)
    outputs['iou'] = ious
    outputs['precisions'] = precisions
    outputs['recalls'] = recalls
    return outputs

def get_metrics(label_id, confusion, VALID_CLASS_IDS: list = None):
    if VALID_CLASS_IDS is not None:
        if not label_id in VALID_CLASS_IDS:
            return float('nan')
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    if VALID_CLASS_IDS is not None:
        not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())
    else:
        not_ignored = [l for l in range(confusion.shape[1]) if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())
    denom = (tp + fp + fn)
    iou = float('nan') if denom == 0 else (float(tp) / denom, tp, denom)
    precision = float('nan') if (tp+fp) == 0 else (float(tp) / (tp+fp), tp, tp+fp)
    recall = float('nan') if (tp+fn) == 0 else (float(tp) / (tp+fn), tp, tp+fn)
    return iou, precision, recall

def evaluate_triplet_recallk(objs_pred, rels_pred, gt_rel, edges, topk_list, topk_each=100, use_clip=False):
    """
    Standard Scene Graph Recall@K evaluation (Scene-Level).
    Adapted from eval_utils_recall.py.
    """
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    # Assumption: rels_pred is already probabilities or logits. 
    # If logits, apply softmax/sigmoid. The reference code applies exp assuming log_softmax.
    # Here we assume rels_pred is consistent with input (check main code).
    # If `indoor_metric.py` passes raw logits, we should softmax them. 
    # For safety, let's assume raw logits and apply softmax if not bounded [0,1].
    if rels_pred.min() < 0 or rels_pred.max() > 1:
         rels_pred = F.softmax(rels_pred, dim=-1)

    all_topk_conf_matrix = None
    all_topk_id = None
    max_k = np.max(topk_list)

    # 1. Collect Candidate Triplets from ALL edges in the scene
    for edge_idx in range(len(edges)):
        edge_from = edges[edge_idx][0]
        edge_to = edges[edge_idx][1]
        
        # [N_rel_cls]
        rel_predictions = rels_pred[edge_idx]
        # [N_obj_cls]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        # Joint Probability: P(s) * P(o) -> [N_obj, N_obj]
        node_score = torch.einsum('n,m->nm', sub, obj)
        # Joint Probability: P(s) * P(o) * P(r) -> [N_obj, N_obj, N_rel]
        conf_matrix = torch.einsum('nl,m->nlm', node_score, rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)

        # Optimization: Take Top-K_EACH for this edge to avoid sorting massive arrays later
        # Default topk_each is 100 (from reference code)
        curr_topk_conf_matrix, curr_topk_conf_id = conf_matrix_1d.topk(min(topk_each, conf_matrix_1d.shape[0]), largest=True)
        
        # Store Edge Index + Flat Class Index
        curr_edge_id = torch.zeros_like(curr_topk_conf_id) + edge_idx
        # shape: [k, 2] -> (edge_idx, flat_cls_idx)
        curr_topk_id = torch.stack([curr_edge_id, curr_topk_conf_id], dim=-1)

        if all_topk_conf_matrix is None:
            all_topk_conf_matrix = curr_topk_conf_matrix
            all_topk_id = curr_topk_id
        else:
            all_topk_conf_matrix = torch.cat([all_topk_conf_matrix, curr_topk_conf_matrix], dim=0)
            all_topk_id = torch.cat([all_topk_id, curr_topk_id], dim=0)

    # 2. Global Sort: Sort ALL candidates from ALL edges
    if all_topk_conf_matrix is None: # Handle empty scene
        return {k: 0.0 for k in topk_list}

    # Take top max_k globally
    num_candidates = all_topk_conf_matrix.shape[0]
    k_global = min(max_k, num_candidates)
    
    global_scores, select_id = all_topk_conf_matrix.topk(k_global, largest=True, sorted=True)
    global_ids = all_topk_id[select_id]

    # 3. Calculate Recall
    correct_number = {k: 0 for k in topk_list}
    
    # Pre-calculate total GTs for this scene
    # gt_rel is list of [sub, obj, [rel_labels]]
    # Note: Reference code sums `len(gt_edge[2])` for total predicates, or `min(1, len)` for triplets.
    # Usually we count total number of ground truth predicates.
    # eval_utils_recall.py -> all_number = sum([len(gt_edge[2]) ...])
    total_gt_count = sum([len(gt_edge[2]) for gt_edge in gt_rel])
    
    if total_gt_count == 0:
        return {k: 0.0 for k in topk_list}

    size_o = objs_pred.shape[1]
    size_r = rels_pred.shape[1]
    
    # Track which GTs have been hit to avoid double counting if necessary 
    # (Reference code logic: `iscompute` dict ensures we don't count same edge multiple times for same K? 
    # Actually reference code counts each correct predicate match).
    
    # We iterate through the global Top-K predictions
    # global_ids: [K, 2] -> (edge_idx, flat_cls_idx)
    
    # Optimize: Map edge_idx to GT for fast lookup
    # gt_rel[edge_idx] = [s_gt, o_gt, [r_gt1, r_gt2...]]
    
    # Because one edge can have multiple GT predicates, we need to track which ones are found.
    # Structure: found_gt[edge_idx] = set(found_pred_classes)
    found_gt_map = {k: {} for k in topk_list} # k -> edge_idx -> set(pred_classes)

    for rank, (edge_idx, flat_id) in enumerate(zip(global_ids[:, 0], global_ids[:, 1])):
        edge_idx = int(edge_idx.item())
        flat_id = int(flat_id.item())
        
        # Decode classes
        # flat_id = s * (N_o * N_r) + o * N_r + r
        # indices = np.unravel_index(flat_id, (size_o, size_o, size_r))
        # PyTorch unravel equivalent:
        r_pred = flat_id % size_r
        rem = flat_id // size_r
        o_pred = rem % size_o
        s_pred = rem // size_o

        # Check GT
        gt_entry = gt_rel[edge_idx]
        s_gt = gt_entry[0]
        o_gt = gt_entry[1]
        r_gts = gt_entry[2] # List of valid predicates

        # Match?
        if s_pred == s_gt and o_pred == o_gt and (r_pred in r_gts):
            # It is a hit. Update counts for all K thresholds that encompass this rank
            curr_rank = rank + 1
            for k in topk_list:
                if curr_rank <= k:
                    # Check if we already found this specific triplet relation for this K
                    if edge_idx not in found_gt_map[k]:
                        found_gt_map[k][edge_idx] = set()
                    
                    if r_pred not in found_gt_map[k][edge_idx]:
                        correct_number[k] += 1
                        found_gt_map[k][edge_idx].add(r_pred)

    # Return fraction for this scene
    return {k: correct_number[k] / total_gt_count for k in topk_list}

def build_gt_triplet(objs_target, rels_target, edges):
    # Construct GT structure: List of [sub_cls, obj_cls, [rel_cls_list]]
    gt_edges = [] 
    objs_target = to_tensor(objs_target)
    rels_target = to_tensor(rels_target) # [N_edges, N_rel_cls] (Multi-hot)
    edges = to_tensor(edges)
    if edges.shape[0] != 2 and edges.shape[1] == 2: edges = edges.t()
    
    num_connections = edges.shape[1]
    for i in range(num_connections):
        idx_sub = edges[0, i].item()
        idx_obj = edges[1, i].item()
        
        s_gt = objs_target[idx_sub].item()
        o_gt = objs_target[idx_obj].item()
        
        # Find all active relations
        r_gt_list = torch.nonzero(rels_target[i]).squeeze(1).tolist()
        
        gt_edges.append([s_gt, o_gt, r_gt_list])
    return gt_edges

class ConfusionMatrix():
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.c_mat = np.zeros([len(self.class_names), len(self.class_names)], dtype=float)
    def get_recall(self):
        return self.c_mat.diagonal().sum() / (self.c_mat.sum() + 1e-6)
    def get_all_metrics(self):
        return get_metrics_all(self.c_mat, [], self.class_names)
    def get_mean_metrics(self):
        all_metrics = self.get_all_metrics()
        mean_metrics = {}
        for metric_name, values in all_metrics.items():
            valid_values = [v[0] for v in values.values() if isinstance(v, tuple) and not math.isnan(v[0])]
            mean_metrics[metric_name] = np.mean(valid_values) if valid_values else 0.0
        return mean_metrics
    def reset(self):
        self.c_mat = np.zeros([len(self.class_names), len(self.class_names)], dtype=float)
    def update(self, pd_tensor, gt_tensor):
        pd_tensor = to_tensor(pd_tensor)
        gt_tensor = to_tensor(gt_tensor)
        pd_tensor = pd_tensor.cpu().numpy()
        gt_tensor = gt_tensor.cpu().numpy()
        for i in range(len(pd_tensor)):
            self.c_mat[gt_tensor[i]][pd_tensor[i]] += 1

class EvalSceneGraphBatch:
    def __init__(self, obj_class_names: list, rel_class_names: list, 
                 multi_rel_threshold: float = 0.5, topk_list: list = [20, 50, 100],
                 iou_threshold: float = 0.25):
        self.obj_class_names = obj_class_names
        
        # 1. 定义类别名称映射 (覆盖默认 ID)
        default_rel_names = [
            'none',          # 0
            'supported by',  # 1
            'attached to',   # 2
            'standing on',   # 3
            'hanging on',    # 4
            'connected to',  # 5
            'part of',       # 6
            'build in'       # 7
        ]
        
        # 逻辑：如果传入的是数字列表，使用默认名称；否则使用传入名称
        if len(rel_class_names) > 0 and str(rel_class_names).isdigit():
             # 长度匹配则替换
             if len(rel_class_names) == len(default_rel_names):
                 self.rel_class_names = default_rel_names
             else:
                 self.rel_class_names = rel_class_names # 长度不匹配，只能用 ID
        else:
            self.rel_class_names = rel_class_names

        self.multi_rel_threshold = multi_rel_threshold
        self.topk_list = sorted(topk_list)
        self.iou_threshold = iou_threshold
        
        # --- Metrics Containers ---
        # 1. VG-Style (Global Ranking, IoU Matching)
        self.results_vg = defaultdict(list)
        self.per_class_hits_vg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # 2. Legacy-Local (Local Ranking per pair) -> 这是能跑出 None 高 Recall 的关键
        self.results_local = defaultdict(list)
        self.per_class_hits_local = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # GT Counters
        self.per_class_counts = defaultdict(int)

        # For R@K (Global)
        self.global_recalls = defaultdict(list)
        
        # For mR@K (Mean Recall per class)
        # Structure: {topk: {rel_cls_id: [1 (hit), 0 (miss), ...]}}
        self.mean_recall_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        self.breakdown_stats = {
            'obj': {k: 0 for k in self.topk_list},
            'pred': {k: 0 for k in self.topk_list},
            'rel': {k: 0 for k in self.topk_list}
        }
        self.object_recall = defaultdict(list)
        self.per_class_hits_object = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.predicate_recall = defaultdict(list)
        self.per_class_hits_predicate = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.triplet_recall = defaultdict(list)
        self.per_class_hits_triplet = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        self.total_gt_rels = 0

    def add(self, data_dict):
        obj_scores = to_tensor(data_dict['obj_scores'])
        obj_boxes = to_tensor(data_dict['obj_boxes']) 
        sub_boxes = to_tensor(data_dict['sub_boxes']) 
        sub_scores = to_tensor(data_dict['sub_scores']) 

        rel_scores = to_tensor(data_dict['rel_scores'])
        # rel_pair_idxs = to_tensor(data_dict['rel_pair_idxs']).long()
        
        gt_obj_labels = to_tensor(data_dict['gt_obj_labels']).long()
        gt_obj_boxes = to_tensor(data_dict['gt_obj_boxes']) 
        gt_rels = to_tensor(data_dict['gt_rels']).long() # [K, 3]

        if gt_rels.shape[0] == 0: return

        # ---------------------------------------------------------------------
        # Pre-process Predictions
        # ---------------------------------------------------------------------
        pred_obj_labels = obj_scores.argmax(1)
        pred_obj_conf = obj_scores.max(1)[0]
        pred_sub_labels = sub_scores.argmax(1)
        pred_sub_conf = sub_scores.max(1)[0]
        
        # Rel Scores: 不屏蔽 Class 0
        rel_class_preds = rel_scores.argmax(1) 
        rel_scores_max = rel_scores.max(1)[0]
        
        # ---------------------------------------------------------------------
        # GT Statistics
        # ---------------------------------------------------------------------
        for cls_id in gt_rels[:, 2]:
            self.per_class_counts[cls_id.item()] += 1
        
        valid_gt_mask = gt_rels[:, 2] > 0
        num_valid_gt = valid_gt_mask.sum().item()
        num_total_gt = gt_rels.shape[0] # 包含 None
        self.total_gt_rels += num_total_gt

        # ===================================================================
        # Metric A: VG-Style (Global Ranking + 3D IoU)
        # ===================================================================
        # 1. Global Score
        # triplet_scores = pred_obj_conf[sub_idxs] * pred_obj_conf[obj_idxs] * rel_scores_max
        triplet_scores = pred_obj_conf * pred_sub_conf * rel_scores_max
        
        # 2. Global Sort
        sorted_idxs = torch.argsort(triplet_scores, descending=True)
        max_k = min(self.topk_list[-1], len(sorted_idxs))
        sorted_idxs = sorted_idxs[:max_k] 
        
        # pred_relations = torch.stack([
        #     sub_idxs[sorted_idxs], obj_idxs[sorted_idxs], rel_class_preds[sorted_idxs]
        # ], dim=1) 
        
        gt_triplets, gt_triplet_boxes = _triplet_tensor(gt_rels, gt_obj_labels, gt_obj_boxes)
        # pred_triplets, pred_triplet_boxes = _triplet_tensor(pred_relations, pred_obj_labels, obj_boxes)
        pred_triplets, pred_triplet_boxes = sub_obj_triplet_tensor(rel_class_preds[sorted_idxs], pred_sub_labels[sorted_idxs], sub_boxes[sorted_idxs], pred_obj_labels[sorted_idxs], obj_boxes[sorted_idxs])

        # 3. Matching
        pred_to_gt_vg = _compute_pred_matches_3d_tensor(
            gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, self.iou_threshold
        )

        # Calculate IoU [Num_GT, Top_K]
        iou_sub = bbox_overlaps_3d(sub_boxes[sorted_idxs], gt_triplet_boxes[:, :7])
        iou_obj = bbox_overlaps_3d(obj_boxes[sorted_idxs], gt_triplet_boxes[:, 7:])
        
        # Spatial Match Mask
        spatial_match = (iou_sub >= self.iou_threshold) & (iou_obj >= self.iou_threshold) # [N_GT, TopK]
        
        # 将所有 Tensor 按分数从高到低重排
        # spatial_match_sorted = spatial_match[:, sorted_idxs] 
        
        # 2. 预测 Label 重排
        p_sub_lbl_sorted = pred_sub_labels[sorted_idxs]   # [N_Pred]
        p_obj_lbl_sorted = pred_obj_labels[sorted_idxs]   # [N_Pred]
        p_rel_lbl_sorted = rel_class_preds[sorted_idxs]   # [N_Pred]

        match_s = intersect_2d_tensor(p_sub_lbl_sorted.unsqueeze(-1), gt_triplets[:, 0].unsqueeze(-1)) # Subject Class Match
        match_o = intersect_2d_tensor(p_obj_lbl_sorted.unsqueeze(-1), gt_triplets[:, 2].unsqueeze(-1)) # Object Class Match
        match_r = intersect_2d_tensor(p_rel_lbl_sorted.unsqueeze(-1), gt_triplets[:, 1].unsqueeze(-1)) # Relation Class Match
        pred_obj_to_gt = [[] for _ in range(p_sub_lbl_sorted.shape[0])]
        # match_obj_only = mask_k & match_s & match_o
        match_obj_only = match_s & match_o
        if match_obj_only.any():
            pred_indices, gt_indices = torch.nonzero(match_obj_only, as_tuple=True)
            
            # Convert to CPU list for iteration in the final metric reduction
            gt_indices = gt_indices.tolist()
            pred_indices = pred_indices.tolist()
            
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                pred_obj_to_gt[pred_idx].append(gt_idx)

        pred_pred_to_gt = [[] for _ in range(p_sub_lbl_sorted.shape[0])]
        # match_pred_only = mask_k & match_r
        match_pred_only = match_r
        if match_pred_only.any():
            pred_indices, gt_indices = torch.nonzero(match_pred_only, as_tuple=True)
            
            # Convert to CPU list for iteration in the final metric reduction
            gt_indices = gt_indices.tolist()
            pred_indices = pred_indices.tolist()
            
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                pred_pred_to_gt[pred_idx].append(gt_idx)

        pred_triplet_to_gt = [[] for _ in range(p_sub_lbl_sorted.shape[0])]
        # match_triplet = mask_k & match_s & match_o & match_r
        match_triplet = spatial_match & match_s & match_o & match_r
        if match_triplet.any():
            pred_indices, gt_indices = torch.nonzero(match_triplet, as_tuple=True)
            
            # Convert to CPU list for iteration in the final metric reduction
            gt_indices = gt_indices.tolist()
            pred_indices = pred_indices.tolist()
            
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                pred_triplet_to_gt[pred_idx].append(gt_idx)

        # -----------------------------------------------------------------------
        # [Step 4] Top-K 循环计算
        # -----------------------------------------------------------------------
        for k in self.topk_list:
        
            if len(pred_to_gt_vg) == 0:
                match = []
            else:
                match = reduce(np.union1d, pred_to_gt_vg[:k]) if k <= len(pred_to_gt_vg) else reduce(np.union1d, pred_to_gt_vg)
            
            valid_hits = 0
            for m in match:
                lbl = gt_rels[int(m), 2].item()
                self.per_class_hits_vg['R'][k][lbl] += 1 
                if lbl > 0: valid_hits += 1 
            
            # 记录 VG-R (Triplet Recall)
            self.results_vg[f'R@{k}'].append(float(valid_hits) / float(num_valid_gt + 1e-6))

            if len(pred_obj_to_gt) == 0:
                match_obj = []
            else:
                match_obj = reduce(np.union1d, pred_obj_to_gt[:k]) if k <= len(pred_obj_to_gt) else reduce(np.union1d, pred_obj_to_gt)
            
            valid_hits = 0
            for m in match_obj:
                lbl = gt_rels[int(m), 2].item()
                self.per_class_hits_object['R'][k][lbl] += 1 
                if lbl > 0: valid_hits += 1 
            
            # 记录 VG-R (Triplet Recall)
            self.object_recall[f'R@{k}'].append(float(valid_hits) / float(num_valid_gt + 1e-6))

            if len(pred_pred_to_gt) == 0:
                match_pred = []
            else:
                match_pred = reduce(np.union1d, pred_pred_to_gt[:k]) if k <= len(pred_pred_to_gt) else reduce(np.union1d, pred_pred_to_gt)
            
            valid_hits = 0
            for m in match_pred:
                lbl = gt_rels[int(m), 2].item()
                self.per_class_hits_predicate['R'][k][lbl] += 1
                if lbl > 0: valid_hits += 1 
            
            # 记录 VG-R (Triplet Recall)
            self.predicate_recall[f'R@{k}'].append(float(valid_hits) / float(num_valid_gt + 1e-6))

            if len(pred_triplet_to_gt) == 0:
                match_triplet = []
            else:
                match_triplet = reduce(np.union1d, pred_triplet_to_gt[:k]) if k <= len(pred_triplet_to_gt) else reduce(np.union1d, pred_triplet_to_gt)
            
            valid_hits = 0
            for m in match_triplet:
                lbl = gt_rels[int(m), 2].item()
                self.per_class_hits_triplet['R'][k][lbl] += 1 
                if lbl > 0: valid_hits += 1 
            
            # 记录 VG-R (Triplet Recall)
            self.triplet_recall[f'R@{k}'].append(float(valid_hits) / float(num_valid_gt + 1e-6))
            # Debug: 如果 Triplet-R 和 VG-R 不一致，说明代码逻辑有差异
            # 通常 VG-R 是正确的基准


        # # VG-ngR Logic ... (略，与之前一致，加上 Class 0 统计)
        # obj_pair_scores = (pred_obj_conf * pred_sub_conf).unsqueeze(1)
        # ng_scores = obj_pair_scores * rel_scores
        # ng_scores_flat = ng_scores.view(-1)
        # ng_sorted_idxs = torch.argsort(ng_scores_flat, descending=True)[:max_k]
        
        # num_classes = rel_scores.shape[1]
        # pair_idxs_sorted = ng_sorted_idxs // num_classes
        # rel_labels_sorted = ng_sorted_idxs % num_classes
        
        # # ng_pred_relations = torch.stack([
        # #     sub_idxs[pair_idxs_sorted], obj_idxs[pair_idxs_sorted], rel_labels_sorted
        # # ], dim=1)
        
        # ng_pred_triplets, ng_pred_triplet_boxes = sub_obj_triplet_tensor(rel_labels_sorted, pred_sub_labels, sub_boxes, pred_obj_labels, obj_boxes)
        # ng_pred_to_gt = _compute_pred_matches_3d_tensor(
        #     gt_triplets, ng_pred_triplets, gt_triplet_boxes, ng_pred_triplet_boxes, self.iou_threshold
        # )
        
        # for k in self.topk_list:
        #     if len(ng_pred_to_gt) == 0:
        #         match = []
        #     else:
        #         match = reduce(np.union1d, ng_pred_to_gt[:k]) if k <= len(ng_pred_to_gt) else reduce(np.union1d, ng_pred_to_gt)
        #     valid_hits = sum([1 for m in match])# if gt_rels[int(m), 2] > 0])
        #     self.results_vg[f'ngR@{k}'].append(float(valid_hits) / float(num_valid_gt + 1e-6))
        #     for m in match:
        #         lbl = gt_rels[int(m), 2].item()
        #         self.per_class_hits_vg['ngR'][k][lbl] += 1

        # ===================================================================
        # Metric B: Legacy-Local (Local Ranking - Mimics 3DSSG util_eva.py)
        # ===================================================================
        # 逻辑：对于每一个 GT 关系（包括 None），找到对应的预测 Pair，
        # 检查 GT Label 在该 Pair 预测分数中的排名。
        # 适用性：PredCls/SGCls (直接索引对应)；SGDet (需要 IoU 匹配)
        
        # 1. 建立 "Predicted Pairs" 的查找表
        # Key: (sub_idx, obj_idx) -> prediction_index
        # 注意：SGDet 可能预测了多个 (s,o)，或者没有。PredCls 是一一对应。
        
        # pred_pair_map = defaultdict(list)
        # for i in range(rel_pair_idxs.shape[0]):
        #     u, v = rel_pair_idxs[i, 0].item(), rel_pair_idxs[i, 1].item()
        #     pred_pair_map[(u, v)].append(i)
            
        # # 2. 遍历每个 GT 关系
        # local_hits = {k: 0 for k in self.topk_list}
        # local_total = 0 # 包含 None 的总数
        
        # 3. 确定是否能直接索引匹配 (PredCls/SGCls)
        # 如果是 PredCls/SGCls，通常 rel_pair_idxs 和 gt_rels 的顺序或索引是一致的
        # 但为了通用，我们尝试通过 Pair Index 查找
        
        # 这里的实现主要针对 PredCls/SGCls 优化 (SGDet 比较复杂，通常 Local Recall 不适用 SGDet)
        
        # for i in range(gt_rels.shape[0]):
        #     gt_s, gt_o, gt_lbl = gt_rels[i, 0].item(), gt_rels[i, 1].item(), gt_rels[i, 2].item()
            
        #     # 在预测中找对应的 Pair
        #     match_pred_indices = []
            
        #     # 策略：如果是 PredCls/SGCls，我们假设 GT Boxes 和 Pred Boxes 是一一对应的 (Index Align)
        #     # 那么直接找 (gt_s, gt_o) 对应的 Pred Pair
        #     if (gt_s, gt_o) in pred_pair_map:
        #         match_pred_indices = pred_pair_map[(gt_s, gt_o)]
            
        #     # 如果是 SGDet，可能需要 IoU 匹配，这里简化处理：
        #     # 如果是 SGDet，Local Recall 意义不大，且很难定义 "None" (没检测到算什么？)
        #     # 所以这里主要服务于 PredCls/SGCls 的对比
            
        #     if not match_pred_indices:
        #         # 没预测这个 Pair -> Miss
        #         continue
                
        #     # 取出该 Pair 的预测分数 (如有多个，取第一个或最高分的)
        #     # util_eva.py 通常处理的是单一预测
        #     pred_idx = match_pred_indices[0] 
        #     scores = rel_scores[pred_idx] # [Num_Classes]
            
        #     # 计算 GT Label 在 Scores 中的排名
        #     # scores 降序排列
        #     sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            
        #     # 找到 gt_lbl 的位置
        #     rank = (sorted_indices == gt_lbl).nonzero().item() + 1
            
        #     # 记录 Hits
        #     for k in self.topk_list:
        #         if rank <= k:
        #             local_hits[k] += 1
        #             self.per_class_hits_local['R'][k][gt_lbl] += 1
            
        #     local_total += 1

        # # 记录 Local Results (包含 None)
        # for k in self.topk_list:
        #     # 分母使用包含 None 的 total
        #     rec = float(local_hits[k]) / float(local_total + 1e-6)
        #     self.results_local[f'R@{k}'].append(rec)

    def gen_text(self):
        txt = "\n" + "=" * 80 + "\n"
        txt += f"Scene Graph Eval (TopK={self.topk_list}, IoU={self.iou_threshold})\n"
        txt += "-" * 80 + "\n"
        
        # 1. 计算各个 Result Dictionary 的均值
        # Helper function to safely calculate mean
        def get_means(res_dict):
            return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in res_dict.items()}

        res_vg = get_means(self.results_vg)
        res_local = get_means(self.results_local)
        
        # 新增的三元组细分指标
        res_obj = get_means(self.object_recall)
        res_pred = get_means(self.predicate_recall) # 注意保持变量名拼写一致
        res_triplet = get_means(self.triplet_recall)
        
        # 2. 计算 Mean Recall (mR)
        # Exclude Class 0 for mean calculation to be standard
        valid_cls = [c for c in self.per_class_counts if self.per_class_counts[c] > 0 and c > 0]
        
        for metric in ['R', 'ngR']:
            for k in self.topk_list:
                per_class_recalls = []
                for c in valid_cls:
                    # 确保 key 存在
                    if c in self.per_class_hits_vg[metric][k]:
                        hits = self.per_class_hits_vg[metric][k][c]
                        total = self.per_class_counts[c]
                        per_class_recalls.append(hits / total)
                    else:
                        per_class_recalls.append(0.0)
                # 将 mR 结果存入 res_vg，key 格式为 'mR@20', 'mngR@20'
                res_vg[f'm{metric}@{k}'] = np.mean(per_class_recalls) if per_class_recalls else 0.0
        metric = 'R'
        for k in self.topk_list:
            object_per_class_recalls = []
            for c in valid_cls:
                # 确保 key 存在
                if c in self.per_class_hits_object[metric][k]:
                    hits = self.per_class_hits_object[metric][k][c]
                    total = self.per_class_counts[c]
                    object_per_class_recalls.append(hits / total)
                else:
                    object_per_class_recalls.append(0.0)
            # 将 mR 结果存入 res_obj，key 格式为 'mR@20', 'mngR@20'
            res_obj[f'mR@{k}'] = np.mean(object_per_class_recalls) if object_per_class_recalls else 0.0
            predicate_per_class_recalls = []
            for c in valid_cls:
                # 确保 key 存在
                if c in self.per_class_hits_predicate[metric][k]:
                    hits = self.per_class_hits_predicate[metric][k][c]
                    total = self.per_class_counts[c]
                    predicate_per_class_recalls.append(hits / total)
                else:
                    predicate_per_class_recalls.append(0.0)
            # 将 mR 结果存入 res_pred，key 格式为 'mR@20', 'mngR@20'
            res_pred[f'mR@{k}'] = np.mean(predicate_per_class_recalls) if predicate_per_class_recalls else 0.0
            triplet_per_class_recalls = []
            for c in valid_cls:
                # 确保 key 存在
                if c in self.per_class_hits_triplet[metric][k]:
                    hits = self.per_class_hits_triplet[metric][k][c]
                    total = self.per_class_counts[c]
                    triplet_per_class_recalls.append(hits / total)
                else:
                    triplet_per_class_recalls.append(0.0)
            # 将 mR 结果存入 res_triplet，key 格式为 'mR@20', 'mngR@20'
            res_triplet[f'mR@{k}'] = np.mean(triplet_per_class_recalls) if triplet_per_class_recalls else 0.0

        # 3. 构建表格头部
        headers = ["Metric"] + [f"@{k}" for k in self.topk_list]
        fmt = "{:<22}" + " | {:<8}" * len(self.topk_list)
        txt += fmt.format(*headers) + "\n"
        txt += "-" * (22 + 11 * len(self.topk_list)) + "\n"
        
        # 4. 定义每一行的配置: (显示名称, 数据源字典, Key前缀)
        # 你的 add 代码中，所有字典的 key 都是 f'R@{k}' (除了 mR/ngR 在 res_vg 中不同)
        rows_config = [
            ("VG-R (Global 3D)",   res_vg,      'R'),
            ("VG-mR (Mean)",       res_vg,      'mR'),
            ("VG-ngR (NoGraph)",   res_vg,      'ngR'),
            ("Local-R (Per-Pair)", res_local,   'R'),
            # 新增指标
            ("Object-R",           res_obj,     'R'),
            ("Object-mR",          res_obj,     'mR'),
            ("Predicate-R",        res_pred,    'R'),
            ("Predicate-mR",       res_pred,    'mR'),
            ("Triplet-R",          res_triplet, 'R'),
            ("Triplet-mR",         res_triplet, 'mR'),
        ]
        
        # 5. 循环输出每一行
        for label, data_dict, prefix in rows_config:
            # 根据前缀和 TopK 构建 Key，例如 'R@20', 'mR@50'
            vals = [data_dict.get(f'{prefix}@{k}', 0.0) for k in self.topk_list]
            vals_str = [f"{v:.4f}" for v in vals]
            txt += fmt.format(label, *vals_str) + "\n"
            
        txt += "-" * (22 + 11 * len(self.topk_list)) + "\n"
        txt += "* VG-R: Standard Global Ranking (Background suppressed).\n"
        txt += "* Local-R: Per-Pair Classification (Includes Background/None).\n"
        
        # 6. Per Class Detail (保持不变)
        txt += f"\nPer-Class Recall @ {self.topk_list[-1]}:\n"
        txt += f"{'ID':<4} | {'Name':<15} | {'Count':<6} | {'Global-R':<8} | {'Local-R':<8}\n"
        txt += "-" * 65 + "\n"
        
        all_present_cls = sorted([c for c in self.per_class_counts if self.per_class_counts[c] > 0])
        
        for c in all_present_cls:
            if 0 <= c < len(self.rel_class_names):
                name = self.rel_class_names[c]
            else:
                name = str(c)
            
            cnt = self.per_class_counts[c]
            
            # Global Recall
            hits_global = self.per_class_hits_vg['R'][self.topk_list[-1]][c]
            r_global = hits_global / cnt if cnt > 0 else 0.0
            
            # Local Recall
            hits_local = self.per_class_hits_local['R'][self.topk_list[-1]][c]
            r_local = hits_local / cnt if cnt > 0 else 0.0
            
            txt += f"{c:<4} | {name:<15} | {cnt:<6} | {r_global:.4f}   | {r_local:.4f}\n"
            
        return txt

        # # Create Table
        # table_data = []
        
        # # 1. Main R@K and mR@K Table
        # table_data.append(["Metric"] + [f"@{k}" for k in self.topk_list])
        # breakdown_rows = []
        # name_map = {'obj': 'Obj-R', 'pred': 'Pred-R', 'rel': 'Rel-R'}
        
        # for key in ['obj', 'pred', 'rel']:
        #     row = [name_map[key]]
        #     for k in self.topk_list:
        #         hits = self.breakdown_stats[key][k]
        #         total = self.total_gt_rels
        #         val = hits / total if total > 0 else 0.0
        #         row.append(f"{val:.4f}")
        #     breakdown_rows.append(row)

        # # Formatting
        # txt = "\n" + "=" * 60 + "\n"
        # txt += "Scene Graph Generation Metrics\n"
        # txt += "-" * 60 + "\n"
        
        # # Original Table
        # table1 = AsciiTable(table_data)
        # txt += table1.table + "\n"
        # txt += "-" * 60 + "\n"
        # txt += "Recall Breakdown (Global)\n"
        # table2 = AsciiTable([table_data[0]] + breakdown_rows)
        # txt += table2.table + "\n"
        txt += "=" * 60 + "\n"
            
        return txt

    def get_result_dict(self):
        res = {}
        for k, v in self.results_vg.items(): res[f'VG_{k}'] = np.mean(v)
        for k, v in self.results_local.items(): res[f'Local_{k}'] = np.mean(v) # Log Local
        
        valid_cls = [c for c in self.per_class_counts if self.per_class_counts[c] > 0 and c > 0]
        for metric in ['R', 'ngR']:
            for k in self.topk_list:
                mR_list = [self.per_class_hits_vg[metric][k][c] / self.per_class_counts[c] for c in valid_cls]
                res[f'VG_m{metric}@{k}'] = np.mean(mR_list) if mR_list else 0.0
                
        return res