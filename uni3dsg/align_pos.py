import torch
import torch. nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_add, scatter_mean
import copy
from mmdet3d.structures import PointData
from mmdet3d.registry import MODELS
from mmdet3d.structures import DepthInstance3DBoxes
from .structures import InstanceData_
import torch
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
from .unidet3d import UniDet3D
from .segdet import mask_matrix_nms
import time

@MODELS.register_module()
class Align3D_POS_single(UniDet3D):
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
                 num_classes,
                 voxel_size,
                 min_spatial_shape,
                 query_thr_seg,
                 query_thr_det,
                 use_superpoints,
                 bbox_by_mask, 
                 target_by_distance,
                 fast_nms,
                 group=1,
                 use_pos_RPE=False,
                 use_gblobs_emb=False,
                 use_ray_loss=False,
                 get_sempan=True,
                 use_sync_bn=True,
                 backbone=None,
                 seg_decoder=None,
                 det_decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(UniDet3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if backbone is not None:
            self.unet = MODELS.build(backbone)
        self.seg_decoder = MODELS.build(seg_decoder)
        self.datasets = det_decoder.datasets
        self.det_decoder = MODELS.build(det_decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes=num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr_seg = query_thr_seg
        self.query_thr_det = query_thr_det
        self.use_superpoints = use_superpoints
        self.bbox_by_mask = bbox_by_mask
        self.target_by_distance = target_by_distance 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_sync_bn = use_sync_bn
        self.fast_nms = fast_nms
        self.get_sempan=get_sempan
        self.group = group
        self.use_pos_PRE=use_pos_RPE
        self.use_gblobs_emb=use_gblobs_emb
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
    def _select_queries_unidet3d_rand(self, x, gt_instances):
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
        sp_centers = []
        updated_instances = []
        query_perc=0.5
        
        for j in range(len(x)*self.group):
            i = j % len(x)
            updated_instance = copy.deepcopy(gt_instances[i])
            if query_perc<1:
                n = (1 - self.query_thr_seg) * torch.rand(1) + self.query_thr_seg
                n = (n * len(x[i])).int()
                '''print("q")
                print(len(x[i]))
                print("n")
                print(n)'''
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                
                queries.append(x[i][ids])
                sp_centers.append(gt_instances[i].sp_centers[ids])
                updated_instance.sp_centers = updated_instance.sp_centers[ids]
                updated_instance.query_masks = updated_instance.sp_masks[:, ids]
                
            else:
                queries.append(x[i])
                sp_centers.append(gt_instances[i].sp_centers)
                updated_instance.query_masks = updated_instance.sp_masks
            updated_instances.append(updated_instance)
        return queries, sp_centers, updated_instances
    def _select_queries_combined(self, x, gt_instances):
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
        sp_centers = []
        updated_instances = []

        for i in range(len(x)):
            updated_instance = copy.deepcopy(gt_instances[i])
            n = (1 - self.query_thr_seg) * torch.rand(1) + self.query_thr_seg
            n = (n * len(x[i])).int()
            if n > self.query_thr_det:
                ids = torch.randperm(len(x[i]))[:self.query_thr_det].to(x[i].device)
                queries.append(x[i][ids])
                sp_centers.append(gt_instances[i].sp_centers[ids])
                updated_instance.sp_centers = updated_instance.sp_centers[ids]
                updated_instance.query_masks = updated_instance.sp_masks[:, ids]
            else:
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                sp_centers.append(gt_instances[i].sp_centers[ids])
                updated_instance.sp_centers = updated_instance.sp_centers[ids]
                updated_instance.query_masks = updated_instance.sp_masks[:, ids]

            updated_instances.append(updated_instance)

        return queries, sp_centers, updated_instances


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
        ori_sp_pts_masks = []
        
        for i in range(len(batch_data_samples)):
            
            datasets_names.append(self.get_dataset(
                            batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            ori_sp_pts_masks.append(copy.deepcopy(gt_pts_seg.sp_pts_mask))
            
            dataset = self.datasets.index(datasets_names[i])
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
            if len(batch_data_samples[i].xyz_mean):
                xyz_mean=torch.vstack(batch_data_samples[i].xyz_mean).to(batch_data_samples[i].gt_instances_3d.labels_3d.device)
                
            else:
                xyz_mean=batch_data_samples[i].xyz_mean
            batch_data_samples[i].gt_instances_3d.xyz_mean=xyz_mean
            batch_data_samples[i].gt_instances_3d.sp_centers = \
                scatter_mean(points[i], gt_pts_seg.sp_pts_mask, dim=0)
            if self.target_by_distance[dataset] and batch_data_samples[i].gt_instances_3d.bboxes_3d is not None:
                if batch_data_samples[i].gt_instances_3d.bboxes_3d.shape[0] > 0:  
                    batch_data_samples[i].gt_instances_3d.sp_masks = \
                        self.get_targets(batch_data_samples[i].gt_instances_3d.sp_centers,
                                        batch_data_samples[i].gt_instances_3d.bboxes_3d,
                                        self.train_cfg.topk)
                
            sp_centers.append(batch_data_samples[i].gt_instances_3d.sp_centers)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

       
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        
        ori_sp_centers=[i.sp_centers for i in sp_gt_instances]
       
        if self.use_gblobs_emb:
            x = self.guass_extract_feat2(
            x, sp_pts_masks, inverse_mapping, batch_offsets, points)
        else:
            x = self.extract_feat(
                x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_centers_queries, sp_gt_instances = \
                    self._select_queries_combined(x, sp_gt_instances)
        
        
        for i in range(len(sp_gt_instances)):   
            sp_gt_instances[i].ori_sp_centers=ori_sp_centers[i % len(x)]
        if self.use_pos_PRE:
            x_seg,x_box = self.seg_decoder(x, queries, sp_centers_queries, ori_sp_centers)
        else:
            x_seg,x_box = self.seg_decoder(x, queries, sp_centers_queries)
             
        

        loss = self.criterion(x_seg,x_box ,sp_gt_instances,  datasets_names,ori_sp_pts_masks,points)
        return loss
    def gaussian_scatter(self, coords, features, superpoints, sigma=1.0):
        """
        高斯加权聚合替代 scatter_mean
        Args:
            coords:     点云坐标 (k, 3)
            features:  点云特征 (k, c)
            superpoints: 超点分组索引 (k,), 如 [0,0,1,1,1] 表示前两点属于超点0,后三点属于超点1
            sigma:      高斯核标准差（可设为可学习参数）
        Returns:
            aggregated_feature: 聚合后的特征 (num_superpoints, c)
        """
        # Step 1: 计算每个超点的坐标中心
        # 使用 scatter_mean 计算每个超点的平均坐标
        center_coords = scatter_mean(coords, superpoints, dim=0)  # (num_superpoints, 3)

        # Step 2: 计算每个点到所属超点中心的距离
        # 将中心坐标根据索引展开到每个点对应的位置
        expanded_centers = center_coords[superpoints]  # (k, 3)
        dist = torch.norm(coords - expanded_centers, dim=1)  # (k,)

        # Step 3: 生成高斯权重 (k, 1)
        weights = torch.exp(-0.5 * (dist ** 2) / (sigma ** 2)).unsqueeze(-1)  # (k, 1)

        # Step 4: 加权聚合（带归一化）
        weighted_features = features * weights  # (k, c)
        sum_weights = scatter_add(weights, superpoints, dim=0)  # (num_superpoints, 1)
        aggregated_feature = scatter_add(weighted_features, superpoints, dim=0)  # (num_superpoints, c)
        
        # 归一化：避免除零错误（添加极小值 epsilon=1e-6）
        aggregated_feature = aggregated_feature / (sum_weights + 1e-6)  # (num_superpoints, c)

        return aggregated_feature

    
    def guass_extract_feat(self, x, superpoints, inverse_mapping, batch_offsets, points):
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
        points=torch.cat(points,dim=0)
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x_grouped=self.group_to_superpoints(points,superpoints,batch_offsets[-1])
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        gbl_=self.gblobs_emb(x_grouped)
        
        out = []
        for i in range(len(batch_offsets) - 1):
            x_i=x[batch_offsets[i]: batch_offsets[i + 1]]
            gbl_i=gbl_[batch_offsets[i]: batch_offsets[i + 1]]
            out_x_i=torch.cat((x_i,gbl_i),dim=1)
            out.append(out_x_i)
        return out
    def guass_extract_feat2(self, x, superpoints, inverse_mapping, batch_offsets, points):
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
        points=torch.cat(points,dim=0)
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = self.gaussian_scatter(
            coords=points,  # 根据逆映射获取原始坐标
            features=x.features[inverse_mapping],
            superpoints=superpoints,
            sigma=0.5  # 可调整或设为可学习参数
        )
        out = []
        for i in range(len(batch_offsets) - 1):
            x_i=x[batch_offsets[i]: batch_offsets[i + 1]]
            out.append(x_i)
        return out



    def group_to_superpoints(self, points,superpoints, superpoints_max):
        """
        Args:
            points:      Tensor of shape [total_pts, 3]
            superpoints: LongTensor of shape [total_pts], 每个值 ≥0
            superpoints_max = superpoints.max() + 1
        Returns:
            grouped: List of (max_label+1) Tensors, grouped[i].shape = [Ki, 3]
        """
        
        grouped = []
        for lab in range(superpoints_max):
            mask = (superpoints == lab)
            grouped.append(points[mask])
        return grouped



    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset
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
                samples. Each Det3DDataSample contains 'pred_instances_3d' 
                with the following keys:
                    - bboxes_3d (Tensor): 3D bounding boxes of detected 
                    instances, shape (num_instances, 6).
                    - scores_3d (Tensor): Classification scores for each 
                    detected instance, shape (num_instances,).
                    - labels_3d (Tensor): Labels of instances, shape 
                    (num_instances,).
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        sp_centers = []
        datasets_names = []
        sp_pts_masks_src = []
        points_src = []
        ori_sp_pts_masks = []


        
        for i in range(len(batch_data_samples)):
            datasets_names.append(self.get_dataset(
                            batch_data_samples[i].lidar_path))
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            ori_sp_pts_masks.append(copy.deepcopy(gt_pts_seg.sp_pts_mask))
            points = batch_inputs_dict['points'][i][:, :3]
            
            #dataset = self.det_decoder.datasets.index(datasets_names[i])
            '''if self.bbox_by_mask[dataset]:
                gt_masks = self.get_gt_inst_masks(gt_pts_seg.pts_instance_mask)
                batch_data_samples[i].gt_instances_3d.bboxes_3d = \
                                            self.get_bboxes_by_masks(gt_masks.T,
                                                                    points)'''
            points_src.append(points)
            sp_centers.append(scatter_mean(points, 
                                           gt_pts_seg.sp_pts_mask, dim=0))
            sp_pts_masks_src.append(gt_pts_seg.sp_pts_mask)
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points']
            )
        
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        if self.use_gblobs_emb:
            x = self.guass_extract_feat2(
                x, sp_pts_masks, inverse_mapping, batch_offsets, points_src)
        else:
            x = self.extract_feat(
                x, sp_pts_masks, inverse_mapping, batch_offsets)

        if self.use_pos_PRE:
            x_seg,x_det = self.seg_decoder(x, x, sp_centers, sp_centers)
        else:
            x_seg,x_det= self.seg_decoder(x, x, sp_centers)
        # x_det = self.det_decoder(x, sp_centers,torch.tensor(batch_offsets), datasets_names)
        # bboxes_det=x_det['bboxes']
        # x_seg = self.seg_decoder(x, x,sp_centers,torch.tensor(batch_offsets), torch.tensor(batch_offsets),bboxes_det)

        '''box_queries,mask_queries=self.align(x_seg,x_det,x, coordinates_real,x,ori_sp_pts_masks,points_src)  
        x_seg['align_mask']= self.get_mask_head(x, mask_queries,self.get_sempan)
        x_det['align_box'] = self.get_det_head(box_queries, sp_centers, datasets_names)'''
        results_list = self.predict_by_feat_seg_det(x_seg,x_det, sp_pts_masks_src, sp_pts_masks,
                                            points_src, datasets_names,self.get_sempan)
        if self.get_sempan:
            for i, data_sample in enumerate(batch_data_samples):
                bboxes, labels, scores = results_list[i]['det']
                res_mask=results_list[i]['seg']
                sem_res=results_list[i]['sem']
                pan_res=results_list[i]['pan']
                pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
                pts_instance_mask = [res_mask[0].cpu().bool().numpy(),pan_res[1].cpu().numpy()]     
                data_sample.pred_pts_seg=PointData(
                    pts_semantic_mask=pts_semantic_mask,
                    pts_instance_mask=pts_instance_mask,
                    mask_instance_labels=res_mask[1].cpu().numpy(),
                    mask_instance_scores=res_mask[2].cpu().numpy())
                data_sample.pred_instances_3d = InstanceData_(
                    bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                    points=batch_inputs_dict['points'][0])
        else:
            for i, data_sample in enumerate(batch_data_samples):
                bboxes, labels, scores = results_list[i]['det']
                res_mask=results_list[i]['seg']
                
                pts_instance_mask = [res_mask[0].cpu().bool().numpy()]     
                data_sample.pred_pts_seg=PointData(
                    pts_instance_mask=pts_instance_mask,
                    mask_instance_labels=res_mask[1].cpu().numpy(),
                    mask_instance_scores=res_mask[2].cpu().numpy())
                data_sample.pred_instances_3d = InstanceData_(
                    bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels,
                    points=batch_inputs_dict['points'][0])

        
            
        return batch_data_samples
        
    
    def gblobs_emb(self, f_list):
        """
        Algorithm 1 严格复现版本。输入一个长度为 N 的 list
        每项是 [K, M]，输出一个长度为 N 的 list
        每项是一个一维张量，形状 [M + M*M]。
        """
        outs = []
        for f in f_list:
            K, M = f.shape
            # 1) mean & center
            f_mean = f.mean(dim=0, keepdim=True)      # [1, M]
            f_loc  = f - f_mean                       # [K, M]

            # 2) covariance
            cov = torch.einsum('km,kn->mn', f_loc, f_loc)  # [M, M]

            # 3) flatten & concat
            d        = f_loc.mean(dim=0, keepdim=True)    # [1, M]
            cov_flat = cov.reshape(1, M*M)                 # [1, M*M]
            g        = torch.cat([d, cov_flat], dim=1)     # [1, M+M*M]

            # squeeze
            outs.append(g)                     # [M+M*M]

        return torch.cat(outs, dim=0)
    def predict_by_feat_seg_det(self, out_seg,out_det, sp_pts_masks, superpoints, points, 
                        datasets_names,get_sempan):
        """Predict bounding boxes and labels from model outputs.

        Args:
            out (dict): A dictionary containing model outputs with the 
                following keys:
                - 'cls_preds': Tensor of shape (n_bboxes, num_classes) 
                containing classification scores for each point.
                - 'bboxes': Tensor of shape (n_bboxes, 7) containing 
                predicted bounding boxes.
            sp_pts_masks (List[Tensor]): A list of superpoint masks.
            points (List[Tensor]): A list of point tensors containing 
                the 3D coordinates of the points being evaluated.

            datasets_names (List[str]): A list of dataset names 
                corresponding to the input samples.

        Returns:
            List[Tuple[DepthInstance3DBoxes, Tensor, Tensor]]: A list containing 
            tuples of predicted bounding boxes and their associated 
            labels and scores.
        """
        seg_inst_res = self.predict_by_feat_instance_seg(
            out_seg, superpoints, self.test_cfg.inst_score_thr)
        
        det_inst_res = self.predict_by_feat_det(out_det, sp_pts_masks, 
                                            points, datasets_names)
        res_dict={}
        if get_sempan:
            sem_res = self.predict_by_feat_semantic(out_seg, superpoints)
            pan_res = self.predict_by_feat_panoptic(out_seg, superpoints)
            res_dict['sem']=sem_res
            res_dict['pan']=pan_res
        res_dict['det']=det_inst_res[0]
        res_dict['seg']=seg_inst_res
        
        return [res_dict]
    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores  = self.predict_by_feat_instance_seg(
            out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map
    def predict_by_feat_instance_seg(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.seg_topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def predict_by_feat_det(self, out, sp_pts_masks, points, 
                        datasets_names):
        """Predict bounding boxes and labels from model outputs.

        Args:
            out (dict): A dictionary containing model outputs with the 
                following keys:
                - 'cls_preds': Tensor of shape (n_bboxes, num_classes) 
                containing classification scores for each point.
                - 'bboxes': Tensor of shape (n_bboxes, 7) containing 
                predicted bounding boxes.
            sp_pts_masks (List[Tensor]): A list of superpoint masks.
            points (List[Tensor]): A list of point tensors containing 
                the 3D coordinates of the points being evaluated.

            datasets_names (List[str]): A list of dataset names 
                corresponding to the input samples.

        Returns:
            List[Tuple[DepthInstance3DBoxes, Tensor, Tensor]]: A list containing 
            tuples of predicted bounding boxes and their associated 
            labels and scores.
        """
        cls_preds = out['cls_preds'][0]
        pred_bboxes = out['bboxes'][0]
        sp_pts_mask = sp_pts_masks[0] 
        point = points[0]
        dataset_name = datasets_names[0]
    
        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        num_classes = scores.shape[1]
        labels = torch.arange(
            num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.det_topk_insts, sorted=True)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, num_classes, rounding_mode='floor')
        pred_bboxes = pred_bboxes[topk_idx]

        fast_nms = self.fast_nms[self.det_decoder.\
                                        datasets.index(dataset_name)]
        iou_thr = self.test_cfg.iou_thr[self.det_decoder.\
                                        datasets.index(dataset_name)]
        nms_bboxes, nms_scores, nms_labels = \
                self._single_scene_multiclass_nms(pred_bboxes,
                                                  scores, 
                                                  labels,
                                                  fast_nms, 
                                                  iou_thr)
        if not self.use_superpoints[
            self.det_decoder.datasets.index(dataset_name)]:
            return [(DepthInstance3DBoxes(
                nms_bboxes, 
                with_yaw=nms_bboxes.shape[1] == 7, 
                box_dim=nms_bboxes.shape[1], 
                origin=(0.5, 0.5, 0.5)),
                nms_labels, nms_scores)]
        else:
            return self.trim_bboxes_by_superpoints(sp_pts_mask, point, 
                                                   nms_bboxes, nms_labels,
                                                   nms_scores)
    
def get_batch_offset(tensors):
    batch_sizes = [tensor.shape[0] for tensor in tensors]

    batch_offsets = torch.cumsum(torch.tensor([0] + batch_sizes), dim=0)
    return batch_offsets
