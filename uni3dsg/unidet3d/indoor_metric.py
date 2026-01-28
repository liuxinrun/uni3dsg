# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
import copy
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from .indoor_eval import indoor_eval, EvalSceneGraphBatch, to_tensor
from .instance_seg_eval import instance_seg_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type

@METRICS.register_module()
class IndoorMetric_(BaseMetric):
    def __init__(self,
                 datasets,
                 datasets_classes,
                 relation_classes=None,
                 eval_with_seg: bool = True,
                 eval_with_ssg: bool = True,
                 vis_dir: str = None,
                 iou_thr: List[float] = [0.25, 0.5],
                 ssg_metrics=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(IndoorMetric_, self).__init__(prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.datasets = datasets
        self.eval_with_seg = eval_with_seg
        self.eval_with_ssg = eval_with_ssg
        self.datasets_classes = datasets_classes 
        self.relation_classes = relation_classes
        self.vis_dir = vis_dir
        
        self.active_ssg_modes = []
        self.ssg_topk_dict = {}
        
        if self.eval_with_ssg and ssg_metrics is not None:
            for metric_cfg in ssg_metrics:
                if metric_cfg['type'] == 'SGRecall':
                    mode = metric_cfg['mode']
                    self.active_ssg_modes.append(mode)
                    self.ssg_topk_dict[mode] = metric_cfg.get('topk', [20, 50, 100])
        
        if self.eval_with_ssg and not self.active_ssg_modes:
             self.active_ssg_modes = ['predcls', 'sgcls']

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                cpu_pred_3d[k] = v.to('cpu') if hasattr(v, 'to') else v
            if self.eval_with_ssg:
                ssg_results = data_sample['ssg_results'] 
                for k, v in ssg_results.items():
                    cpu_pred_3d[k] = v.to('cpu') if hasattr(v, 'to') else v
            if self.eval_with_seg:
                 pred_mask_3d = data_sample['pred_pts_seg']
                 for k, v in pred_mask_3d.items():
                    cpu_pred_3d[k] = v.to('cpu') if hasattr(v, 'to') else v
            cpu_pred_3d['dataset'] = self.get_dataset(data_sample['lidar_path'])
            eval_ann_info = data_sample['eval_ann_info'] 
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = [[] for _ in self.datasets]
        pred_results = [[] for _ in self.datasets]
        evaluators = {}

        for eval_ann, single_pred_results in results:
            d_name = single_pred_results['dataset']
            idx = self.datasets.index(d_name)
            ann_infos[idx].append(eval_ann)
            
            if 'gt_bboxes_3d' in eval_ann:
                single_pred_results['gt_boxes'] = torch.cat([
                    eval_ann["gt_bboxes_3d"].gravity_center,
                    eval_ann["gt_bboxes_3d"].tensor[:, 3:]], dim=1)
            if 'gt_labels_3d' in eval_ann:
                single_pred_results['gt_labels'] = eval_ann['gt_labels_3d']
            
            pred_results[idx].append(single_pred_results)

            if self.eval_with_ssg and 'sgdet_branch' in single_pred_results:
                sgcls_branch = single_pred_results['sgcls_branch']
                predcls_branch = single_pred_results['predcls_branch']
                sgdet_branch = single_pred_results['sgdet_branch']

                
                gt_labels = to_tensor(single_pred_results['gt_labels']).long()
                gt_rels_matrix = to_tensor(eval_ann['relationships']).long()
                if d_name not in evaluators:
                    obj_classes = self.datasets_classes[idx]
                    if self.relation_classes is not None:
                        rel_classes = self.relation_classes
                    elif sgdet_branch is not None:
                        rel_classes = [str(i) for i in range(sgdet_branch['rel_scores'].shape[1])] 
                    evaluators[d_name] = {}
                    for mode in self.active_ssg_modes:
                        k_list = self.ssg_topk_dict.get(mode, [20, 50, 100])
                        evaluators[d_name][mode] = EvalSceneGraphBatch(obj_classes, rel_classes[0], topk_list=k_list)


                gt_rel_map = defaultdict(list)
                if gt_rels_matrix.ndim > 1:
                    for r in gt_rels_matrix:
                        gt_rel_map[(int(r[0]), int(r[1]))].append(int(r[2]))
                


                if 'predcls' in self.active_ssg_modes and predcls_branch is not None:
                    
                    data_predcls = {
                        'obj_scores': predcls_branch['obj_scores'],
                        'obj_boxes': predcls_branch['obj_boxes'],
                        'sub_scores': predcls_branch['sub_scores'],
                        'sub_boxes': predcls_branch['sub_boxes'],
                        'rel_scores': predcls_branch['rel_scores'],
                        'gt_obj_labels': gt_labels,
                        'gt_obj_boxes': single_pred_results['gt_boxes'],
                        'gt_rels': gt_rels_matrix
                    }
                    evaluators[d_name]['predcls'].add(data_predcls)

                if 'sgcls' in self.active_ssg_modes and sgcls_branch is not None:
                    data_sgcls = {
                        'obj_scores': sgcls_branch['obj_scores'],
                        'obj_boxes': sgcls_branch['obj_boxes'],
                        'sub_scores': sgcls_branch['sub_scores'],
                        'sub_boxes': sgcls_branch['sub_boxes'],
                        'rel_scores': sgcls_branch['rel_scores'],
                        'gt_obj_labels': gt_labels,
                        'gt_obj_boxes': single_pred_results['gt_boxes'],
                        'gt_rels': gt_rels_matrix
                    }
                    evaluators[d_name]['sgcls'].add(data_sgcls)

                if 'sgdet' in self.active_ssg_modes and sgdet_branch is not None:
                    data_sgdet = {
                        'obj_scores': sgdet_branch['obj_scores'],
                        'obj_boxes': sgdet_branch['obj_boxes'],
                        'sub_scores': sgdet_branch['sub_scores'],
                        'sub_boxes': sgdet_branch['sub_boxes'],
                        'rel_scores': sgdet_branch['rel_scores'],
                        'gt_obj_labels': gt_labels,
                        'gt_obj_boxes': torch.cat([eval_ann["gt_bboxes_3d"].gravity_center, eval_ann["gt_bboxes_3d"].tensor[:, 3:]], dim=1),
                        'gt_rels': gt_rels_matrix
                    }
                    evaluators[d_name]['sgdet'].add(data_sgdet)

        ret_dict = {}
        box_type_3d, box_mode_3d = get_box_type(self.dataset_meta.get('box_type_3d', 'depth'))
        for i in range(len(self.datasets)):
            d_name = self.datasets[i]
            if len(ann_infos[i]) > 0:
                ret_dict[d_name] = indoor_eval(ann_infos[i], pred_results[i], self.iou_thr, self.datasets_classes[i], logger=logger, box_mode_3d=box_mode_3d)

                if self.eval_with_ssg and d_name in evaluators:
                    print_log(f"\nEvaluating Scene Graph for {d_name}...", logger=logger)
                    for mode in self.active_ssg_modes:
                        if mode in evaluators[d_name]:
                            report = evaluators[d_name][mode].gen_text()
                            print_log(f"--- {mode.upper()} ---", logger=logger)
                            print_log(report, logger=logger)
                            
                            res_dict = evaluators[d_name][mode].get_result_dict()
                            for k, v in res_dict.items():
                                ret_dict[f'{d_name}/{mode}_{k}'] = v
        
        if self.eval_with_seg:
             seg_inst_result = self.compute_seg_metrics(logger, results)
             ret_dict.update(seg_inst_result)
             
        return ret_dict

    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset
    
    def map_inst_markup(self, pts_semantic_mask, pts_instance_mask, valid_class_ids, num_stuff_cls):
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1
        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        return pts_semantic_mask, pts_instance_mask

    def compute_seg_metrics(self, logger, results):
        vaild_idxs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        num_stuff_cls=2
        for eval_ann, single_pred_results in results:
            sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                vaild_idxs[num_stuff_cls:], num_stuff_cls)
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)           
            pred_instance_masks_inst_task.append(torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(torch.tensor(single_pred_results['mask_instance_labels']))
            pred_instance_scores.append(torch.tensor(single_pred_results['mask_instance_scores']))
        
        ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task, gt_instance_masks_inst_task,
                pred_instance_masks_inst_task, pred_instance_labels, pred_instance_scores,
                valid_class_ids=vaild_idxs[num_stuff_cls:],
                class_labels=self.dataset_meta['classes'], logger=logger)
        return ret_inst
    
