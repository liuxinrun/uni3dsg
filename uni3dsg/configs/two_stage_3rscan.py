_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['unidet3d'])


classes_3rscan = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
        'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

relation_classes_3rscan = ['none', 'supported by', 'attached to', 'standing on', 'hanging on', 'connected to',  'part of', 'build in']

# model settings
num_channels=32
voxel_size=0.02

model = dict(
    type='Two_STAGE',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=[False],
    target_by_distance=[True],
    use_superpoints=[False],
    fast_nms=[True],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_3rscan],
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['3rscan'],
        angles=[False]),
    bbox_roi_extractor=dict(
            type='Single3DRoIPointExtractor2',
            roi_layer=dict(type='RoIPointPool3d', num_sampled_points=512)),
    criterion=dict(
        type='UniDet3DCriterion',
            datasets=['3rscan'],
            datasets_weights=[1],
            bbox_loss_simple=dict(
                type='UniDet3DAxisAlignedIoULoss',
                mode='diou',
                reduction='none'),
            bbox_loss_rotated=dict(
                type='UniDet3DRotatedIoU3DLoss',
                mode='diou',
                reduction='none'),
            matcher=dict(
                type='UniMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='BboxCostJointTraining', 
                            weight=2.0,
                            loss_simple=dict(
                                type='UniDet3DAxisAlignedIoULoss',
                                mode='diou',
                                reduction='none'),
                            loss_rotated=dict(
                                type='UniDet3DRotatedIoU3DLoss',
                                mode='diou',
                                reduction='none'))]),
            loss_weight=[0.5, 1.0],
            non_object_weight=0.1,
            topk=[3],
            iter_matcher=True),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18,
        up_sp_thr=0.81,
        topk_insts=1000,
        score_thr=0,
        iou_thr=[0.55]))

# 3rscan dataset settings
data_root_3rscan = 'dataset/3rscan'
dataset_type_3rscan = 'ThreeRScan_'
data_prefix_3rscan = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='superpoints')

train_pipeline_3rscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=True,
         with_bbox_3d=True,
         with_sp_mask_3d=True),
    dict(type='PointSample_', 
         num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'elastic_coords', 'gt_bboxes_3d', 
              'gt_labels_3d', 'sp_pts_mask'])
]
test_pipeline_3rscan = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_',
         with_label_3d=False,
         with_bbox_3d=False,
         with_sp_mask_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample_', num_points=100000),
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]


# run settings
train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset_',
        datasets=[dict(
                    type=dataset_type_3rscan,
                    ann_file='3rscan_infos_train.pkl',
                    partition=0.15,
                    data_prefix=data_prefix_3rscan,
                    data_root=data_root_3rscan,
                    pipeline=train_pipeline_3rscan,
                    test_mode=False)]
                    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset_',
        datasets= \
                [dict(
                    type=dataset_type_3rscan,
                    ann_file='3rscan_infos_val.pkl',
                    data_prefix=data_prefix_3rscan,
                    data_root=data_root_3rscan,
                    pipeline=test_pipeline_3rscan,
                    test_mode=True)]# + \
                    ))

test_dataloader = val_dataloader

load_from = 'work_dirs/unidet3d_1xb8_3rscan/epoch_512.pth'

test_evaluator = dict(
    type='IndoorMetric_', 
    datasets=['3rscan'],
    datasets_classes=[classes_3rscan],
    eval_with_seg=False,
    iou_thr=[0.25, 0.5],
    ssg_metrics=[
        dict(
            type='SGRecall',
            mode='predcls', 
            topk=[1, 3, 5, 10, 20, 50, 100] 
        ),
        dict(
            type='SGRecall',
            mode='sgcls', 
            topk=[1, 3, 5, 10, 20, 50, 100]
        ),
        dict(
            type='SGRecall',
            mode='sgdet', 
            topk=[1, 3, 5, 10, 20, 50, 100]
        )
    ])

val_evaluator = test_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001*2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024,
    dynamic_intervals=[(1, 16), (1024 - 16, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
