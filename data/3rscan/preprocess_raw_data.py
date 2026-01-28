import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
import sys
import os
import argparse
import glob
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import trimesh
from collections import defaultdict

from utils import read_objmesh, point_indices_from_group, point_indices_from_group_160, find_neighbors, SAMPLE_METHODS, get_label, getLabelNameMapping, read_txt_to_list

# 配置前缀
CLOUD_FILE_PFIX = 'mesh.refined.v2'
AGGREGATIONS_FILE_PFIX = 'semseg.v2.json'
SEGMENTS_FILE_PFIX = 'mesh.refined.0.010000.segs.v2.json'

SUPPORT_TYPE_RELATIONSHIPS = [
    'none', 'supported by', 'attached to',
    'standing on', 'hanging on', 'connected to', 'part of', 'build in'
]

# SUPPORT_TYPE_RELATIONSHIPS = [
#     "none", "supported by", "left", "right", "front", "behind",
#     "close by", "inside", "bigger than", "smaller than",
#     "higher than", "lower than", "same symmetry as", "same as",
#     "attached to", "standing on", "lying on", "hanging on",
#     "connected to", "leaning against", "part of", "belonging to",
#     "build in", "standing in", "cover", "lying in", "hanging in"
# ]

ALL_RELATIONSHIPS = [
    "none", "supported by", "left", "right", "front", "behind",
    "close by", "inside", "bigger than", "smaller than",
    "higher than", "lower than", "same symmetry as", "same as",
    "attached to", "standing on", "lying on", "hanging on",
    "connected to", "leaning against", "part of", "belonging to",
    "build in", "standing in", "cover", "lying in", "hanging in"
]

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def compute_superpoint_geometric_feats(points, sp_mask):
    """
    输入:
        points: [N, 3] 原始点云坐标
        sp_mask: [N] 每个点所属的超点 ID (0 ~ K-1)
    输出:
        sp_geo_feats: [K, 4] -> [linearity, planarity, sphericity, verticality]
        sp_normals: [K, 3] -> PCA 估算的法向量
    """
    # 确保是 numpy array
    points = np.array(points)
    sp_mask = np.array(sp_mask)
    
    unique_ids = np.unique(sp_mask)
    if len(unique_ids) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    num_sp = unique_ids.max() + 1
    
    # 初始化特征
    # 4维: [Linearity, Planarity, Sphericity, Verticality]
    sp_geo_feats = np.zeros((num_sp, 4), dtype=np.float32)
    sp_normals = np.zeros((num_sp, 3), dtype=np.float32)
    
    # Z轴向量 (用于计算垂直度)
    z_axis = np.array([0, 0, 1])

    for uid in unique_ids:
        if uid < 0: continue # 忽略背景
        
        # 获取该超点的所有点
        mask = (sp_mask == uid)
        pts = points[mask]
        
        if len(pts) < 5: 
            continue # 点太少，特征保持 0
        
        # 1. PCA 计算
        # 中心化
        pts_c = pts - pts.mean(axis=0)
        # 协方差矩阵
        cov = np.dot(pts_c.T, pts_c) / (len(pts) - 1)
        # 特征值分解 (升序: e[0] <= e[1] <= e[2])
        # 注意：np.linalg.eigh 适用于对称矩阵，且返回特征值升序排列
        try:
            e, v = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        
        # 防止分母为0
        e_sum = e.sum() + 1e-6
        e_norm = e / e_sum
        # l1, l2, l3 对应 lambda_min, lambda_mid, lambda_max
        
        # 2. 定义几何形态指标 (Intrinsic Shape Features)
        # Linearity (线状): (l3 - l2) / l3 (或者 e[2]-e[1] / e[2])
        # Planarity (面状): (l2 - l1) / l3 
        # Scattering (球状): l1 / l3
        # 使用原始特征值计算比率
        denom = e[2] + 1e-6
        linearity = (e[2] - e[1]) / denom
        planarity = (e[1] - e[0]) / denom
        sphericity = e[0] / denom
        
        # 3. 法向量与垂直度 (Verticality)
        # 最小特征值对应的特征向量即为法向量
        normal = v[:, 0] 
        # 确保法向量朝上 (可选，为了由一致性)
        if normal[2] < 0: normal = -normal
        
        # Verticality: 法向量与 Z 轴夹角的余弦值
        # 1.0 = 水平面 (Normal is vertical), 0.0 = 垂直墙面
        verticality = np.abs(np.dot(normal, z_axis))
        
        sp_geo_feats[uid] = [linearity, planarity, sphericity, verticality]
        sp_normals[uid] = normal

    return sp_geo_feats, sp_normals

def get_raw2id_160_mapping(pth_mapping="", pth_160=""):
    """
    构建一个从 原始字符串标签 到 160类ID 的字典
    """
    # 1. 获取 160 类别的名称列表
    # 假设 pth_160 指向包含 160 个类名的 txt 文件
    if pth_160 == "":
        pth_160 = "./data/3rscan/meta_data/classes160.txt" # 请确保 define 被正确引入
    
    # 读取 160 类名并排序 (与你提供的代码逻辑保持一致)
    names_160 = sorted(read_txt_to_list(pth_160))
    
    # 构建 规范化名称 -> ID 的映射 (ID 从 1 开始)
    # 比如: {'wall': 1, 'floor': 2, ...}
    name_to_id_160 = {name: i + 1 for i, name in enumerate(names_160)}
    
    # 2. 获取 原始标签 -> 规范化标签 的映射
    # 这里调用你提供的 getLabelNameMapping 函数
    # NameScan3R528 格式应该是: {'dark leather chair': 'chair', ...}
    # _, _, _, _, _ = getLabelNames(pth_mapping) # 可能不需要这个，但保持调用顺序
    NameScan3R528, _, _, _, _ = getLabelNameMapping(pth_mapping)
    
    # 3. 构建最终映射: 原始标签字符串 -> 160 ID
    raw_str_to_160_id = {}
    
    for raw_name, canonical_name in NameScan3R528.items():
        # 如果规范化名称在 160 类列表中，则记录 ID，否则为 0 (忽略)
        if canonical_name in name_to_id_160:
            raw_str_to_160_id[raw_name] = name_to_id_160[canonical_name]
        else:
            raw_str_to_160_id[raw_name] = 0
            
    return raw_str_to_160_id

def read_transform_matrix(Scan3RJson_PATH):
    rescan2ref = {}
    with open(Scan3RJson_PATH, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene.get("scans", []):
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.array(scans["transform"]).reshape(4, 4).T
    return rescan2ref

def get_reference_dic(Scan3RJson_PATH):
    meta_data = json.load(open(Scan3RJson_PATH))
    reference_dic = {}
    for record in meta_data:
        reference = record['reference']
        reference_dic[reference] = reference
        if 'scans' not in record:
            continue
        for scan in record['scans']:
            reference_dic[scan['reference']] = reference
    return reference_dic

def compute_bbox_from_points(pts, label_id):
    """pts: (N,3)"""
    xmin, ymin, zmin = np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])
    xmax, ymax, zmax = np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    w, l, h = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    return np.array([cx, cy, cz, w, l, h, label_id], dtype=np.float32)

def handle_process(scene_path, label_map, output_path, labels_pd,
                   train_scenes, val_scenes, test_scenes, relationships_master_data):
    scene_id = scene_path.rstrip('/').split('/')[-1]

    gt_ply_path = os.path.join(scene_path, 'labels.instances.annotated.v2.ply')
    obj_path = os.path.join(scene_path, f'{CLOUD_FILE_PFIX}.obj')
    aggregations_file = os.path.join(scene_path, f'{AGGREGATIONS_FILE_PFIX}')
    segments_file = os.path.join(scene_path, f'{SEGMENTS_FILE_PFIX}')

    # 几何对齐矩阵
    rot_matrix = rescan2ref.get(scene_id, np.identity(4))
    ref_scene_id = reference_dic.get(scene_id, scene_id)
    ref_rot_matrix = reference_axis_align_matrix_dic[ref_scene_id]

    # split 判定
    if scene_id in train_scenes:
        split_name = 'train'
    elif scene_id in val_scenes:
        split_name = 'val'
    elif scene_id in test_scenes:
        split_name = 'test'
    else:
        print('*', scene_id, 'does not exist in [train, val, test] that have seg files')
        return

    print('Processing:', scene_id, 'in', split_name)

    # 读取 mesh 点云
    pointcloud, faces_array = read_objmesh(obj_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]

    # 旋转到参考对齐坐标
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    r_points = np.dot(rot_matrix, r_points)       # rescan->ref 几何
    r_points = np.dot(ref_rot_matrix, r_points)   # ref 的 axis align
    aligned_pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    # 初始化语义与实例标签
    labelled_pc = np.zeros((pointcloud.shape[0], 1), dtype=np.int32) - 1  # -1: unannotated
    instance_ids = np.zeros((pointcloud.shape[0], 1), dtype=np.int32) - 1

    # 容器：仅在“有效对象”后登记
    objects_id = []                 # 存有效的原始 objectId 列表
    instance_ids_map = {}           # objectId -> 连续 idx
    instance_bboxes = []            # list of [7]
    aligned_instance_bboxes = []    # list of [7]

    seg_groups = []
    seg_indices = None

    if os.path.isfile(aggregations_file):
        # segIndices
        with open(segments_file) as f:
            segments = json.load(f)
            seg_indices = np.array(segments['segIndices'])

        # segGroups
        with open(aggregations_file) as f:
            aggregation = json.load(f)
            seg_groups = list(aggregation['segGroups'])
    else:
        # test 可能没有 aggregation
        seg_groups = []
        seg_indices = np.zeros((points.shape[0],), dtype=np.int32)

    # ==========================================================================
    # [调用新增逻辑] 计算 Superpoints 的无监督几何特征
    # ==========================================================================
    # 使用对齐后的点云坐标 (aligned_pointcloud[:, :3]) 和 seg_indices
    # 注意：seg_indices 必须已加载。
    
    print(f"  Computing superpoint geometric features for {scene_id}...")
    sp_geo_feats, sp_normals = compute_superpoint_geometric_feats(
        aligned_pointcloud[:, :3], 
        seg_indices
    )

    # 遍历每个 group，构造有效对象
    valid_idx = 0
    mapping_dict_160 = get_raw2id_160_mapping(config.label_map_file) 
    for group in seg_groups:
        if label_map == '3rscan20':
            segment_points, aligned_segment_points, p_inds, label_id = point_indices_from_group(
                pointcloud, aligned_pointcloud, seg_indices, group, labels_pd
            )
        elif label_map == '3rscan160':
            segment_points, aligned_segment_points, p_inds, label_id = point_indices_from_group_160(
                pointcloud, aligned_pointcloud, seg_indices, group, mapping_dict_160
            )

        # 必须是有效对象（有点）
        if len(segment_points) == 0:
            continue

        # bbox 基于 segment_points
        bbox = compute_bbox_from_points(segment_points[:, :3], label_id)
        instance_bboxes.append(bbox)

        # 对齐 bbox 优先用 aligned_segment_points，否则退回原始（保证一一对应）
        if len(aligned_segment_points) > 0:
            aligned_bbox = compute_bbox_from_points(aligned_segment_points[:, :3], label_id)
            # 标注该区域的 instance id
            instance_ids[p_inds] = valid_idx
        else:
            # 即便没有 aligned points，仍然保留（用原始点计算的 bbox 作为退路）
            aligned_bbox = compute_bbox_from_points(segment_points[:, :3], label_id)

        aligned_instance_bboxes.append(aligned_bbox)

        # 语义标签
        labelled_pc[p_inds] = label_id

        # 仅此时登记 id 映射
        oid = group["id"]  # 注意：3RScan aggregation 的 objectId
        objects_id.append(oid)
        instance_ids_map[oid] = valid_idx

        valid_idx += 1

    # 转 numpy
    if len(instance_bboxes) == 0:
        # 若无对象，填充占位（train/val 也可能存在空的场景），但关系与邻居自然为空
        instance_bboxes = np.zeros((0, 7), dtype=np.float32)
        aligned_instance_bboxes = np.zeros((0, 7), dtype=np.float32)
    else:
        instance_bboxes = np.stack(instance_bboxes, axis=0).astype(np.float32)
        aligned_instance_bboxes = np.stack(aligned_instance_bboxes, axis=0).astype(np.float32)
        
    # 基本一致性检查
    if (np.any(np.isnan(pointcloud)) or not np.all(np.isfinite(pointcloud))):
        raise ValueError('nan in pointcloud')

    num_boxes = aligned_instance_bboxes.shape[0]


    points_for_search = aligned_pointcloud[:, :3]
    search_labels = instance_ids.flatten()
    raw_neighbors_dict = find_neighbors(
        points_for_search,
        search_labels,
        SAMPLE_METHODS.RADIUS, # 或者改为 SAMPLE_METHODS.RADIUS 如果你的 utils 支持
        receptive_field=0.5 
    )
    neighbor_triples = []
    
    # raw_neighbors_dict 的 key 和 value 已经是 valid_idx (即 0, 1, 2...)
    # -1 代表背景 (unknown/unannotated)，需要过滤
    for src_idx, tgt_idxs in raw_neighbors_dict.items():
        if src_idx == -1: continue
        
        for tgt_idx in tgt_idxs:
            if tgt_idx == -1: continue
            if src_idx == tgt_idx: continue
            
            # 这里的 src_idx, tgt_idx 已经是最终 output 的索引，直接保存
            neighbor_triples.append([src_idx, tgt_idx])

    existing_rel_pairs = set()

    # 关系：优先用 reference 场景键（兼容 3DSSG 的 reference 组织）
    relationship_triples = []
    

    # 关系三元组（带关系类型）
    if split_name != 'test':
        scan_key = reference_dic.get(scene_id, scene_id)
        scan_relationships = relationships_master_data.get(scan_key, None)
        if scan_relationships:
            for rel in scan_relationships:
                if len(rel) < 4: continue
                src_oid, tgt_oid, _, rel_name = rel
                
                # 将 JSON 的 objectId 转为我们的 valid_idx
                if src_oid not in instance_ids_map or tgt_oid not in instance_ids_map:
                    continue
                
                s_idx = instance_ids_map[src_oid]
                t_idx = instance_ids_map[tgt_oid]
                if s_idx == t_idx: continue
                if label_map == '3rscan160' and rel_name in ALL_RELATIONSHIPS:
                    if rel_name == 'none': continue # 显式忽略 json 里的 none (如果有的话)
                    
                    rel_idx = ALL_RELATIONSHIPS.index(rel_name)
                    relationship_triples.append([s_idx, t_idx, rel_idx])
                    existing_rel_pairs.add((s_idx, t_idx))
                if label_map == '3rscan20' and rel_name in SUPPORT_TYPE_RELATIONSHIPS:
                    if rel_name == 'none': continue # 显式忽略 json 里的 none (如果有的话)
                    
                    rel_idx = SUPPORT_TYPE_RELATIONSHIPS.index(rel_name)
                    relationship_triples.append([s_idx, t_idx, rel_idx])
                    existing_rel_pairs.add((s_idx, t_idx))

        # B. 生成 None 关系 (Neighbors - Existing Relations)
        if label_map == '3rscan160':
            none_rel_idx = ALL_RELATIONSHIPS.index('none')
        elif label_map == '3rscan20':
            none_rel_idx = SUPPORT_TYPE_RELATIONSHIPS.index('none')
        
        for n_pair in neighbor_triples:
            s_idx, t_idx = n_pair
            # 如果是邻居，但没有正样本关系，则标记为 none
            if (s_idx, t_idx) not in existing_rel_pairs:
                relationship_triples.append([s_idx, t_idx, none_rel_idx])
                existing_rel_pairs.add((s_idx, t_idx)) # 防止重复
    # print(relationship_triples)

    # 最终严格断言（有助于在预处理阶段发现问题）
    assert all(0 <= s < num_boxes and 0 <= t < num_boxes and s != t
               for s, t, *rest in relationship_triples), \
        f"relationship_triples contains invalid indices in {scene_id},relationship_triples are {relationship_triples}, num boxes are {num_boxes}"
    assert all(0 <= s < num_boxes and 0 <= t < num_boxes and s != t
               for s, t in neighbor_triples), \
        f"neighbor_triples contains invalid indices in {scene_id}, neighbor_triples are {neighbor_triples}, num boxes are {num_boxes}"

    # 输出目录与保存
    out_scene_dir = os.path.join(output_path, f'{scene_id}')
    create_dir(out_scene_dir)
    output_prefix = os.path.join(out_scene_dir, f'{scene_id}')

    # 保存核心产物
    np.save(output_prefix + '_aligned_vert.npy', aligned_pointcloud[:, :6])
    np.save(output_prefix + '_sem_label.npy', labelled_pc.astype(np.int32))
    np.save(output_prefix + '_ins_label.npy', instance_ids.astype(np.int32))
    np.save(output_prefix + '_aligned_bbox.npy', aligned_instance_bboxes.astype(np.float32))
    np.save(output_prefix + '_superpoints.npy', seg_indices.astype(np.int32) if seg_indices is not None else np.zeros((points.shape[0],), dtype=np.int32))
    # [保存新增特征]
    np.save(output_prefix + '_sp_geo_feats.npy', sp_geo_feats)
    np.save(output_prefix + '_sp_normals.npy', sp_normals)

    if split_name != 'test':
        np.save(output_prefix + '_relationships.npy', np.array(relationship_triples, dtype=np.int64))
        np.save(output_prefix + '_neighbors.npy', np.array(neighbor_triples, dtype=np.int64))

    # 可选：简要日志
    print(f"- {scene_id}: boxes={num_boxes}, rel={len(relationship_triples)}, neigh={len(neighbor_triples)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='3RScan_raw',
                        help='Path to the 3RScan dataset containing scene folders')
    parser.add_argument('--output_root', default='preprocessed_3rscan_160',
                        help='Output path where processed data will be located')
    parser.add_argument('--label_map_file',
                        default='meta_data/3RScan.v2_Semantic-Classes-Mapping.csv',
                        help='path to 3RScan semantic class mapping csv')
    parser.add_argument('--label_map', default='3rscan20',
                        help='label map to use: 3rscan20 or 3rscan160')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='The number of parallel workers')
    parser.add_argument('--splits_path', default='meta_data/split',
                        help='Where the txt files with the train/val/test splits live')
    parser.add_argument('--relationships_file',
                        default='data/3rscan/meta_data/relationships.json',
                        help='Path to the master relationships.json file from 3DSSG')
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(config.label_map_file, sep=',', header=1)

    # Load train/val/test splits
    with open(os.path.join(config.splits_path, '3rscan_train.txt')) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(os.path.join(config.splits_path, '3rscan_val.txt')) as val_file:
        val_scenes = val_file.read().splitlines()
    with open(os.path.join(config.splits_path, '3rscan_test.txt')) as test_file:
        test_scenes = test_file.read().splitlines()

    # 3RScan meta：rescan->ref 变换与 reference 归并
    META_FILE = 'data/3rscan/meta_data/3RScan.json'
    rescan2ref = read_transform_matrix(META_FILE)
    reference_dic = get_reference_dic(META_FILE)

    # reference 的 axis align 矩阵
    with open('data/3rscan/meta_data/reference_axis_align_matrix.pkl', 'rb') as f:
        reference_axis_align_matrix_dic = pickle.load(f)

    # 加载 3DSSG 主关系
    with open(config.relationships_file, 'r') as f:
        master_data = json.load(f)['scans']
    # 键：scan 或 reference scan
    relationships_master_data = {s['scan']: s['relationships'] for s in master_data}

    os.makedirs(config.output_root, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(p for p in glob.glob(os.path.join(config.dataset_root, '*'))
                if os.path.isdir(p) and not p.endswith('.txt'))

    # Preprocess data with multiprocessing
    print('Processing scenes...')
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(pool.map(
        handle_process, scene_paths,
        repeat(config.label_map),
        repeat(config.output_root), repeat(labels_pd),
        repeat(train_scenes), repeat(val_scenes),
        repeat(test_scenes), repeat(relationships_master_data)
    ))