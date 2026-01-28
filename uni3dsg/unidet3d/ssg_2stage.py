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
from .criterion_ssg import SSGCriterion

@MODELS.register_module()
class Two_STAGE(UniDet3D):
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
        self.criterion = SSGCriterion()
        self.bbox_roi_extractor=MODELS.build(bbox_roi_extractor)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.use_superpoints = use_superpoints
        self.bbox_by_mask = bbox_by_mask
        self.target_by_distance = target_by_distance 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_sync_bn = use_sync_bn
        self.fast_nms = fast_nms
        self.require_overlap = False     
        self.test_overlap = False        
        self.use_gt_box = True           
        self.use_gt_object_label = True  
        self.fg_thres = 0.2
        self.num_sample_per_gt_rel = 4
        self.batch_size_per_image = 256
        self.positive_fraction = 0.5
        self.num_pos_per_img = 128
        self.gnn_model = TripletGCNModel(
            num_layers=5,
            dim_node=35,
            dim_edge=70,
            dim_hidden=256,
            with_bn=False)
        rel_class_weights = torch.ones(8, device='cuda')
        rel_class_weights[0] = 0.1 
        self.obj_head = nn.Sequential(
                                nn.Linear(35, 512), nn.ReLU(),
                                nn.Linear(512, 256), nn.ReLU(),
                                nn.Linear(256, 21))
        self.rel_head = nn.Sequential(
                                nn.Linear(70, 512), nn.ReLU(),
                                nn.Linear(512, 256), nn.ReLU(),
                                nn.Linear(256, 8))
        self.proposal_matcher = SSGMatcher(0.8, 0.2)
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        if self.use_sync_bn:
            self.output_layer = spconv.SparseSequential(
                torch.nn.SyncBatchNorm(num_channels, eps=1e-4, momentum=0.1),
                torch.nn.ReLU(inplace=True))
        else:
            self.output_layer = spconv.SparseSequential(
                torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
                torch.nn.ReLU(inplace=True))
            
    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass    

    def collate(self, points, elastic_points=None):
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

        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])


        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

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

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        
        return x

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        gt_relationships, gt_neighbors = [], []
        sp_pts_masks = []
        sp_centers = []
        
        if batch_inputs_dict.get('elastic_coords') is not None:
            points = [(point - point.min(0)[0]) * self.voxel_size for point in \
                batch_inputs_dict['elastic_coords']]
            shifts = [point.min(0)[0] * self.voxel_size for point in \
                batch_inputs_dict['elastic_coords']]
        else:
            points = [point[:, :3] - point[:, :3].min(0)[0] for point in \
                batch_inputs_dict['points']]
            shifts = [point[:, :3].min(0)[0] for point in \
                batch_inputs_dict['points']]

        datasets_names = []
        for i in range(len(batch_data_samples)):
            datasets_names.append(self.get_dataset(
                            batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            dataset = self.decoder.datasets.index(datasets_names[i])
            if self.bbox_by_mask[dataset]:
                gt_masks = self.get_gt_inst_masks(gt_pts_seg.pts_instance_mask)
                batch_data_samples[i].gt_instances_3d.bboxes_3d = \
                                            self.get_bboxes_by_masks(gt_masks.T,
                                                                    points[i])
            else:
                center = batch_data_samples[i].gt_instances_3d.\
                                    bboxes_3d.gravity_center - \
                                    shifts[i]
                bboxes = torch.cat((center,
                                    batch_data_samples[i].gt_instances_3d.\
                                    bboxes_3d.tensor[:, 3:]),
                                    dim=1)
                batch_data_samples[i].gt_instances_3d.bboxes_3d = \
                    DepthInstance3DBoxes(
                        bboxes, 
                        with_yaw=batch_data_samples[i].gt_instances_3d.\
                                                    bboxes_3d.with_yaw, 
                        box_dim=bboxes.shape[1], origin=(0.5, 0.5, 0.5))
            
            batch_data_samples[i].gt_instances_3d.sp_centers = \
                scatter_mean(points[i], gt_pts_seg.sp_pts_mask, dim=0)
            if self.target_by_distance[dataset]:
                batch_data_samples[i].gt_instances_3d.sp_masks = \
                    self.get_targets(batch_data_samples[i].gt_instances_3d.\
                                        sp_centers,
                                     batch_data_samples[i].gt_instances_3d.\
                                        bboxes_3d,
                                     self.train_cfg.topk)
            sp_centers.append(batch_data_samples[i].gt_instances_3d.sp_centers)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            gt_relationships.append(batch_data_samples[i].relationships)
            gt_neighbors.append(batch_data_samples[i].neighbors)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        with torch.no_grad():
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))
            sp_pts_masks = torch.hstack(sp_pts_masks)
            feats = self.extract_feat(
                x, sp_pts_masks, inverse_mapping, batch_offsets)
            x = scatter_mean(feats.features[inverse_mapping], sp_pts_masks, dim=0)
            out, point_features = [], []
            for i in range(len(batch_offsets) - 1):
                point_features.append(feats.features[inverse_mapping][batch_offsets[i]: batch_offsets[i + 1]])
                out.append(x[batch_offsets[i]: batch_offsets[i + 1]])

            queries, sp_centers_queries, sp_gt_instances = \
                        self._select_queries(out, sp_gt_instances)
            det_box = self.decoder(queries, sp_centers_queries, datasets_names)

        det_boxes, det_scores, det_labels = self.predict_nms(det_box, datasets_names)
        proposals, rel_labels, rel_pair_idxs, rel_binarys, proposal_labels = \
            self.detect_relsample(det_boxes, det_labels, sp_gt_instances, gt_relationships)

        node_features_list, edge_feature_list = [], []
        obj_preds, rel_preds = [], []
        for i in range(len(proposals)):
            current_box_preds = proposals[i] 
            boxes = _bbox_to_loss(current_box_preds)

            current_features = point_features[i]  
            current_coordinates = coordinates[coordinates[:, 0] == i, 1:] * self.voxel_size
            rois = current_box_preds.unsqueeze(0) 
            roi_features = self.bbox_roi_extractor(current_features.unsqueeze(0), current_coordinates.unsqueeze(0), 1, rois)
            node_features = torch.mean(roi_features, dim=1)
            object_idxs = rel_pair_idxs[i][:, 0]
            subject_idxs = rel_pair_idxs[i][:, 1]
            node_edges = torch.stack([object_idxs, subject_idxs], dim=0) 
            object_features = node_features[object_idxs]
            subject_features = node_features[subject_idxs]
            rel_features = torch.cat([object_features, subject_features], dim=-1) 

            edge_feature_list.append(rel_features)
            node_features_list.append(node_features)
            gcn_obj_feature, gcn_rel_feature = self.gnn_model(node_features, rel_features, node_edges)
            obj_pred = self.obj_head(gcn_obj_feature)
            rel_pred = self.rel_head(gcn_rel_feature)
            rel_scores = F.softmax(rel_pred, dim=-1)
            obj_preds.append(obj_pred)
            rel_preds.append(rel_pred)
        
        
        loss = self.criterion(obj_preds, rel_preds, rel_labels, proposal_labels)

        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples
                with post-processing.

        Args:
            batch_inputs_dict (dict): A dictionary containing model inputs, 
                which must include 'points' key.
            batch_data_samples (List[:obj:Det3DDataSample]): A list of Data 
                Samples. Each Data Sample includes information such as
                superpoints (gt_pts_seg.sp_pts_mask).

        Returns:
            List[:obj:Det3DDataSample]: Detection results for the input 
                samples. Each Det3DDataSample contains:
                - pred_instances_3d: Detection results
                - ssg_results: Scene graph generation results
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        sp_centers = []
        datasets_names = []
        sp_pts_masks_src = []
        points_src = []
        gt_relationships, gt_neighbors = [], []
        for i in range(len(batch_data_samples)):
            datasets_names.append(self.get_dataset(
                            batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            points = batch_inputs_dict['points'][i][:, :3]
            points_src.append(points)
            sp_centers.append(scatter_mean(points, 
                                           gt_pts_seg.sp_pts_mask, dim=0))
            sp_pts_masks_src.append(gt_pts_seg.sp_pts_mask)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)
            gt_relationships.append(batch_data_samples[i].relationships)
            gt_neighbors.append(batch_data_samples[i].neighbors)

        coordinates, features, inverse_mapping, spatial_shape, shift = self.collate_shift(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        feats = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        x = scatter_mean(feats.features[inverse_mapping], sp_pts_masks, dim=0)
        out, point_features = [], []
        for i in range(len(batch_offsets) - 1):
            point_features.append(feats.features[inverse_mapping][batch_offsets[i]: batch_offsets[i + 1]])
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])

        gt_boxes_list, gt_labels_list, gt_rel_pairs_list = [], [], []
        
        for ds in batch_data_samples:
            gt = ds.eval_ann_info
            rois = torch.cat([gt["gt_bboxes_3d"].gravity_center,
                              gt["gt_bboxes_3d"].tensor[:, 3:]], dim=1).to(point_features[0].device)
            gt_boxes_list.append(rois)
            gt_labels_list.append(torch.tensor(gt["gt_labels_3d"]).to(rois.device))

            if len(ds.relationships) > 0:
                gt_rel_pairs_list.append(torch.tensor(ds.relationships, device=rois.device)) # [M, 2]
            else:   
                num_nodes = len(rois)
                subj, obj = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
                edge_index = torch.stack([subj.flatten(), obj.flatten()], dim=0)
                edge_index = edge_index[:, edge_index[0] != edge_index[1]].to(rois.device) # 去自环
                gt_rel_pairs_list.append(edge_index.T) # [M, 2]

        gt_results = self._inference_sg_branch(
            coordinates, point_features, shift, 
            gt_boxes_list, gt_rel_pairs_list
        )

        det_box_out = self.decoder(out, sp_centers, datasets_names)
        det_boxes_nms, det_scores_nms, det_labels_nms = self.predict_nms(det_box_out, datasets_names)
        

        det_rel_pairs_list = []
        det_valid_mask = [] 
        
        for i, boxes in enumerate(det_boxes_nms):
            if len(boxes) > 1:
                num_nodes = len(boxes)
                subj, obj = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
                edge_index = torch.stack([subj.flatten(), obj.flatten()], dim=0)
                edge_index = edge_index[:, edge_index[0] != edge_index[1]].to(boxes.device)
                det_rel_pairs_list.append(edge_index.T)
                det_valid_mask.append(True)
            else:
                det_rel_pairs_list.append(torch.zeros((0, 2), dtype=torch.long, device=boxes.device))
                det_valid_mask.append(False)

        det_results = self._inference_sg_branch(
            coordinates, point_features, shift,
            det_boxes_nms, det_rel_pairs_list
        )

        results_list = self.predict_by_feat(det_box_out, sp_pts_masks_src, 
                                            points_src, datasets_names)
        
        for i, data_sample in enumerate(batch_data_samples):
            bboxes, labels, scores = results_list[i]
            data_sample.pred_instances_3d = InstanceData_(
                bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                points=batch_inputs_dict['points'][i])

            final_sub_labels = gt_labels_list[i][gt_rel_pairs_list[i][:, 0]]
            final_obj_labels = gt_labels_list[i][gt_rel_pairs_list[i][:, 1]]

            num_cls = gt_results['obj_preds'][i].shape[-1] - 1 
            
            final_sub_scores = torch.zeros((len(gt_rel_pairs_list[i][:, 0]), num_cls), device=boxes.device)
            final_obj_scores = torch.zeros((len(gt_rel_pairs_list[i][:, 1]), num_cls), device=boxes.device)

            final_sub_scores.scatter_(1, final_sub_labels.unsqueeze(1), 1.0)
            final_obj_scores.scatter_(1, final_obj_labels.unsqueeze(1), 1.0)

            if det_boxes_nms[i].shape[0] > 0:
                final_bboxes = torch.cat((det_boxes_nms[i], torch.zeros_like(det_boxes_nms[i][:, :1])), dim=1)
            else:
                final_bboxes = torch.zeros((0, det_boxes_nms[i].shape[1] + 1), device=det_boxes_nms[i].device)

            ssg_result = {
                'sgcls_branch': {
                    'sub_boxes': gt_boxes_list[i][gt_rel_pairs_list[i][:, 0]], # Subject boxes
                    'sub_scores': F.softmax(gt_results['obj_preds'][i][gt_rel_pairs_list[i][:, 0]], dim=-1)[:, :-1], # Model predicted scores for SGCls
                    'obj_boxes': gt_boxes_list[i][gt_rel_pairs_list[i][:, 1]],   # Object boxes
                    'obj_scores': F.softmax(gt_results['obj_preds'][i][gt_rel_pairs_list[i][:, 1]], dim=-1)[:, :-1], # Model predicted scores for SGCls
                    'rel_scores': F.softmax(gt_results['rel_scores'][i], dim=-1),
                    'gt_boxes': gt_boxes_list[i], 
                    'gt_labels': gt_labels_list[i]
                },
                'predcls_branch': {
                    'sub_boxes': gt_boxes_list[i][gt_rel_pairs_list[i][:, 0]], # Subject boxes
                    'sub_scores': final_sub_scores, # One-Hot GT labels as scores
                    'obj_boxes': gt_boxes_list[i][gt_rel_pairs_list[i][:, 1]],   # Object boxes
                    'obj_scores': final_obj_scores, # One-Hot GT labels as scores
                    'rel_scores': F.softmax(gt_results['rel_scores'][i], dim=-1),
                    'gt_boxes': gt_boxes_list[i], # Pass GT boxes for sanity check
                    'gt_labels': gt_labels_list[i]
                },
                
                # --- Det Branch (SGDet) ---
                'sgdet_branch': {
                    'obj_scores': F.softmax(det_results['obj_preds'][i][det_rel_pairs_list[i][:, 1]], dim=-1)[:, :-1],
                    'obj_boxes': final_bboxes[det_rel_pairs_list[i][:, 1]],
                    'sub_scores': F.softmax(det_results['obj_preds'][i][det_rel_pairs_list[i][:, 0]], dim=-1)[:, :-1],
                    'sub_boxes': final_bboxes[det_rel_pairs_list[i][:, 0]],
                    'rel_scores': F.softmax(det_results['rel_scores'][i], dim=-1),
                    'gt_boxes': gt_boxes_list[i], # Pass GT boxes for sanity check
                    'gt_labels': gt_labels_list[i]
                }
            }
            data_sample.ssg_results = ssg_result
            
            if hasattr(data_sample, 'gt_instances_3d'):
                data_sample.relationships = data_sample.gt_instances_3d.relationships

        return batch_data_samples
    
    def _inference_sg_branch(self, coordinates, point_features, shift, boxes_list, pairs_list):
        obj_preds = []
        rel_preds = []
        rel_scores = []
        
        for i, (boxes, pairs) in enumerate(zip(boxes_list, pairs_list)):
            if len(boxes) == 0 or len(pairs) == 0:
                obj_preds.append(torch.zeros((0, 21), device=point_features[0].device))
                rel_preds.append(torch.zeros((0, 8), device=point_features[0].device))
                rel_scores.append(torch.zeros((0, 8), device=point_features[0].device))
                continue

            current_features = point_features[i].unsqueeze(0)
            current_coordinates = coordinates[coordinates[:, 0] == i, 1:].unsqueeze(0) * self.voxel_size + shift[i]

            roi_features = self.bbox_roi_extractor(current_features, current_coordinates, 1, boxes)
            node_features = torch.mean(roi_features, dim=1) # [N, C]

            subject_idxs = pairs[:, 0]
            object_idxs = pairs[:, 1]
            node_edges = torch.stack([subject_idxs, object_idxs], dim=0) # [2, M]

            object_features = node_features[object_idxs]
            subject_features = node_features[subject_idxs]
            rel_features = torch.cat([subject_features, object_features], dim=-1)

            gcn_obj_feature, gcn_rel_feature = self.gnn_model(node_features, rel_features, node_edges)

            op = self.obj_head(gcn_obj_feature)
            rp = self.rel_head(gcn_rel_feature)
            rs = F.softmax(rp, dim=-1)
            
            obj_preds.append(op)
            rel_preds.append(rp)
            rel_scores.append(rs)
            
        return {'obj_preds': obj_preds, 'rel_preds': rel_preds, 'rel_scores': rel_scores}

    def predict_nms(self, out, datasets_names):
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(len(out['cls_preds'])):
            cls_preds = out['cls_preds'][i]
            pred_bboxes = out['bboxes'][i]
            dataset_name = datasets_names[i]

            scores = F.softmax(cls_preds, dim=-1)[:, :-1]
            num_classes = scores.shape[1]
            labels = torch.arange(
                num_classes,
                device=scores.device).unsqueeze(0).repeat(
                    len(cls_preds), 1).flatten(0, 1)
            topk = min(len(scores.flatten(0, 1)), self.test_cfg.topk_insts)
            scores, topk_idx = scores.flatten(0, 1).topk(
                topk, sorted=True)
            labels = labels[topk_idx]

            topk_idx = torch.div(topk_idx, num_classes, rounding_mode='floor')
            pred_bboxes = pred_bboxes[topk_idx]


            fast_nms = self.fast_nms[self.decoder.datasets.index(dataset_name)]
            iou_thr = self.test_cfg.iou_thr[self.decoder.datasets.index(dataset_name)]
            nms_bbox, nms_score, nms_label = \
                    self._single_scene_multiclass_nms(pred_bboxes,
                                                    scores, 
                                                    labels,
                                                    fast_nms, 
                                                    iou_thr)
            nms_bboxes.append(nms_bbox)
            nms_scores.append(nms_score)
            nms_labels.append(nms_label)
        return nms_bboxes, nms_scores, nms_labels
    

    def detect_relsample(self, proposals, pred_labels, targets, relationships):
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        proposal_labels = []
        for idx, (proposal, pred_label, target, relationship) in enumerate(zip(proposals, pred_labels, targets, relationships)):
            device = proposal.device
            prp_box = _bbox_to_loss(proposal[:, :6])
            prp_lab = pred_label.long()
            tgt_box = _bbox_to_loss(torch.cat((target.bboxes_3d.gravity_center, 
                                      target.bboxes_3d.tensor[:, 3:] if \
                                      target.bboxes_3d.with_yaw else \
                                      target.bboxes_3d.tensor[:, 3:6]),
                                      dim=1))
            tgt_lab = target.labels_3d
            tgt_rel = relationship
            n_gt = len(tgt_box)
            tgt_rel_matrix = torch.zeros((n_gt, n_gt), dtype=int, device=device) # [tgt, tgt]
            for rel in tgt_rel:
                subject, object_, relation = rel
                tgt_rel_matrix[subject, object_] = relation
            # IoU matching
            ious = AxisAlignedBboxOverlaps3D()(tgt_box, prp_box, is_aligned=False)  # [tgt, prp]
            matched_idxs = self.proposal_matcher(ious)
            matched_targets = tgt_lab[matched_idxs.clamp(min=0)]
            matched_targets[matched_idxs < 0] = torch.tensor(20, device=device)
            proposal_labels.append(matched_targets)
            is_match = (ious > self.fg_thres)
            prp_self_iou = AxisAlignedBboxOverlaps3D()(prp_box, prp_box, is_aligned=False) 
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match, rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2]) # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2]) # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys, proposal_labels

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1,1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1,-1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab]*prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,1)
            fg_rel_i = cat((prp_head_idxs.view(-1,1), prp_tail_idxs.view(-1,1), fg_labels), dim=-1).to(torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score 
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(-1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = np.random.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)
        
        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility>0).view(-1,2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1,1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=device)

        return cat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

class SSGMatcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold):
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the point clouds "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the point clouds "
                    "during training")
        matched_vals, matches = match_quality_matrix.max(dim=0)

        all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = SSGMatcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = SSGMatcher.BETWEEN_THRESHOLDS

        self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches
    
    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):

        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]