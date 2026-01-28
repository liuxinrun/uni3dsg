import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
from torch import Tensor
import numpy as np

from mmcv.ops import nms3d, nms3d_normal
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.models import Base3DDetector
from mmdet3d.models.layers.box3d_nms import aligned_3d_nms
from mmdet3d.structures import rotation_3d_in_axis, AxisAlignedBboxOverlaps3D, bbox_overlaps_3d

from .criterion import _bbox_to_loss
from .structures import InstanceData_
from .unidet3d import UniDet3D
import torch.nn as nn
from .gcn import TripletGCNModel
from .criterion_ssg import EGTR3DCriterion

@MODELS.register_module()
class UNI3DSG(UniDet3D):
    """UniDet3D for unifed 3D object detection.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select min(query_thr, n_queries) queries
            for training and testing.
        use_superpoints (bool): Flag to indicate whether to use superpoints
            for improved detection.
        bbox_by_mask (bool): Whether to derive bounding boxes from masks.
        target_by_distance (bool): Whether to use targets based on distance 
            to bbox center.
        fast_nms (bool): Flag for using fast Non-Maximum Suppression.
        use_sync_bn (bool, optional): Flag to use synchronized 
            batch normalization. Defaults to True.
        backbone (ConfigDict, optional): Config dict of the backbone. 
            Defaults to None.
        decoder (ConfigDict, optional): Config dict of the decoder. 
            Defaults to None.
        criterion (ConfigDict, optional): Config dict of the criterion. 
            Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters. 
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process 
            config of :class:BaseDataPreprocessor.
            It usually includes:
                - ``pad_size_divisor``
                - ``pad_value``
                - ``mean``
                - ``std``.
        init_cfg (dict or ConfigDict, optional): The config to control the 
            initialization. Defaults to None.
    """
    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 min_spatial_shape,
                 query_thr,
                 use_superpoints,
                 bbox_by_mask, 
                 target_by_distance,
                 fast_nms,
                 query_rel_thr=300,
                 bbox_roi_extractor=None,
                 use_sync_bn=True,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(UniDet3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if backbone is not None:
            self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.query_rel_thr = query_rel_thr
        self.use_superpoints = use_superpoints
        self.bbox_by_mask = bbox_by_mask
        self.target_by_distance = target_by_distance 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_sync_bn = use_sync_bn
        self.fast_nms = fast_nms
        self.fg_thres = 0.2
        self.num_sample_per_gt_rel = 4
        self.batch_size_per_image = 256
        self.positive_fraction = 0.5
        self.num_pos_per_img = 128
        self._init_layers(in_channels, num_channels)
            
    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass  

    def _select_rel_queries(self, x, sp_centers, sp_gt_instances=None):
        """Select queries for the training pass.

        Args:
            x (List[Tensor]): A list of tensors of length `batch_size`, 
                where each tensor has the shape (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): A list of ground truth 
                instances of length `batch_size`, where each instance may 
                contain:
                    - labels of shape (n_gts_i,)
                    - sp_masks of shape (n_gts_i, n_points_i).

        Returns:
            Tuple[List[Tensor], List[Tensor], List[InstanceData_]]:
                - queries (List[Tensor]): A list of queries of length 
                `batch_size`, where each query has the shape 
                (n_queries_i, n_channels).
                - sp_centers (List[Tensor]): A list of tensors representing 
                spatial centers for the selected queries.
                - updated_gt_instances (List[InstanceData_]): A list of ground 
                truth instances (same length as `gt_instances`), 
                each updated with query_masks of shape (n_gts_i, n_queries_i).
        """
        queries = []
        rel_sp_centers = []
        for i in range(len(x)):
            if len(x[i]) > self.query_rel_thr:
                ids = torch.randperm(len(x[i]))[:self.query_rel_thr].to(x[i].device)
                queries.append(x[i][ids])
                rel_sp_centers.append(sp_centers[i][ids])
                if self.training:
                    sp_gt_instances[i].rel_query_masks = sp_gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                rel_sp_centers.append(sp_centers[i])
                if self.training:
                    sp_gt_instances[i].rel_query_masks = sp_gt_instances[i].sp_masks
        return queries, rel_sp_centers

    def collate_shift(self, points, elastic_points=None):
        """Collate a batch of points into a sparse tensor.

        Args:
            points (List[Tensor]): A batch of point tensors. Each tensor
                should contain points in the format (N, 3 + num_features),
                where N is the number of points.
            elastic_points (List[Tensor], optional): A batch of transformed
                point tensors (if any) after elastic point augmentation. 
                Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: 
                - coordinates (Tensor): The sparse tensor coordinates after 
                quantization and normalization.
                - features (Tensor): The features corresponding to the points.
                - inverse_mapping (Tensor): A mapping of points to their 
                indices in the original tensor.
                - spatial_shape (Tensor): The spatial shape of the sparse tensor,
                clipped to the minimum spatial shape.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
            shifts = [p[:, :3].min(0)[0] for p in points]
            
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
            shifts = [el_p[:, :3].min(0)[0] for el_p in elastic_points]

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape, shifts
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        sp_centers = [] # 原始 superpoints 中心
        datasets_names = []
        sp_pts_masks_src = []
        points_src = []
        gt_neighbors = []
        sp_gt_instances = []
        
        for i in range(len(batch_data_samples)):
            datasets_names.append(self.get_dataset(batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            points = batch_inputs_dict['points'][i][:, :3]
            points_src.append(points)
            
            curr_sp_center = scatter_mean(points, gt_pts_seg.sp_pts_mask, dim=0)
            sp_centers.append(curr_sp_center)

            sp_pts_masks_src.append(gt_pts_seg.sp_pts_mask)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)
            gt_neighbors.append(batch_data_samples[i].neighbors)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)

        x = self.extract_feat(x, sp_pts_masks, inverse_mapping, batch_offsets)

        out_det = self.decoder(x, x, x, sp_centers, sp_centers, sp_centers, datasets_names)

        det_results = []
        for i in range(len(batch_data_samples)):
            sub_final_bboxes = out_det['sub_pred_bboxes'][i]
            obj_final_bboxes = out_det['obj_pred_bboxes'][i]
            sub_probs = F.softmax(out_det['sub_cls_preds'][i], dim=-1) 
            obj_probs = F.softmax(out_det['obj_cls_preds'][i], dim=-1)
            obj_probs_valid = obj_probs[:, :-1]
            sub_probs_valid = sub_probs[:, :-1]
            rel_probs = F.softmax(out_det['rel_preds'][i], dim=-1)

            sub_final_bboxes = torch.cat((sub_final_bboxes, torch.zeros_like(sub_final_bboxes[:, :1])), dim=1)
            obj_final_bboxes = torch.cat((obj_final_bboxes, torch.zeros_like(obj_final_bboxes[:, :1])), dim=1)
            
                

            det_result = {'sub_boxes': sub_final_bboxes,
                      'sub_scores': sub_probs_valid,
                      'obj_boxes': obj_final_bboxes,
                      'obj_scores': obj_probs_valid,
                      'rel_scores': rel_probs}
            det_results.append(det_result)

        gt_branch_results = [None] * len(batch_data_samples)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.ssg_results = {
                'sgdet_branch': det_results[i],
                'sgcls_branch': gt_branch_results[i],
                'predcls_branch': gt_branch_results[i]
            }

            if not hasattr(data_sample, 'relationships') and hasattr(data_sample, 'gt_instances_3d'):
                 data_sample.relationships = data_sample.gt_instances_3d.relationships

        results_list = self.predict_by_feat(out_det, sp_pts_masks_src, 
                                            points_src, datasets_names)
        for i, data_sample in enumerate(batch_data_samples):
            bboxes, labels, scores = results_list[i]
            data_sample.pred_instances_3d = InstanceData_(
                bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                points=batch_inputs_dict['points'][i])

        return batch_data_samples