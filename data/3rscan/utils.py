import os
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pandas as pd
from enum import Enum
import open3d as o3d

NYU40_Label_Names = [
    'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'blinds',
'desk',
'shelves',
'curtain',
'dresser',
'pillow',
'mirror',
'floor mat',
'clothes',
'ceiling',
'books',
'refridgerator',
'television',
'paper',
'towel',
'shower curtain',
'box',
'whiteboard',
'person',
'night stand',
'toilet',
'sink',
'lamp',
'bathtub',
'bag',
'otherstructure',
'otherfurniture',
'otherprop',
]

SCANNET20_Label_Names = [
'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'desk',
'curtain',
'refridgerator',
'shower curtain',
'toilet',
'sink',
"bathtub",
'otherfurniture',
]
# from scannet200_constants import *

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)                
    return arr

def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    return normals

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.array([f[0] for f in plydata["face"].data])
        return vertices, faces

def read_objmesh(filepath):
    v, vt, vn, faceV, uvIDs, mtlfile = loadOBJ(filepath)
    dirpath = os.path.dirname(filepath)
    mtlpath = os.path.join(dirpath, mtlfile)
    mtl_images = read_mtl_file(mtlpath)
    
    if len(faceV.keys()) != 1:
        #print("obj file have multiple textures")
        print("This code is not for OBJ file which have multiple textures")
        print("it may not work well")
        exit()

    vc = texture_to_vertex_color(vt, uvIDs, mtl_images, dirpath)
    vertex_data = concat_obj_data(v, vc, vn)

    if len(vn) == 0:
        vertex_plyfile = convert_vertex_data_to_plyfile_format(vertex_data, False)
    else:
        vertex_plyfile = convert_vertex_data_to_plyfile_format(vertex_data, True)

    face_plyfile = convert_face_data_to_plyfile_format(faceV)

    plydata = PlyData(
                [
                    PlyElement.describe(
                        vertex_plyfile, 'vertex'),
                    PlyElement.describe(face_plyfile, 'face')
                ],
                text=True, byte_order='='
            )

    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.array([f[0] for f in plydata["face"].data])
        # print('vertices:', vertices)
        # print('faces:', faces)
        # exit()
        return vertices, faces        


def save_plymesh(vertices, faces, filename, verbose=True, with_label=True):
    """Save an RGB point cloud as a PLY file.

    Args:
      points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
          the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
    """
    assert vertices.ndim == 2
    if with_label:
        if vertices.shape[1] == 7:
            python_types = (float, float, float, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4')]

        if vertices.shape[1] == 8:
            python_types = (float, float, float, int, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4'), ('instance_id', 'u4')]

        if vertices.shape[1] == 10:
            python_types = (float, float, float, int, int, int, float, float, float, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('label', 'u4')]
                        
        if vertices.shape[1] == 11:
            python_types = (float, float, float, int, int, int, float, float, float, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('label', 'u4'), ('instance_id', 'u4')]            
    else:
        if vertices.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (vertices.shape[0], 3))
            vertices = np.hstack((vertices, gray_concat))
        elif vertices.shape[1] == 6:
            python_types = (float, float, float, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1')]
        elif vertices.shape[1] == 9:
            python_types = (float, float, float, int, int, int, float, float, float)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]            
        else:
            pass

    vertices_list = []
    for row_idx in range(vertices.shape[0]):
        cur_point = vertices[row_idx]
        vertices_list.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices_list, dtype=npy_types)
    elements = [PlyElement.describe(vertices_array, 'vertex')]

    if faces is not None:
        faces_array = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
        faces_array['vertex_indices'] = faces
        elements += [PlyElement.describe(faces_array, 'face')]

    # Write
    PlyData(elements).write(filename)

    if verbose is True:
        print('Saved point cloud to: %s' % filename)

def point_indices_from_group(points, aligned_points, seg_indices, group, labels_pd):
    group_segments = np.array(group['segments'])
    label = group['label']

    label_ids = labels_pd[labels_pd['Label'] == label]['Unnamed: 2']
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
    assert label_id != 0
    if label_id == 0:
        
        print(f'label: {label}')

    point_IDs = np.where(np.isin(seg_indices, group_segments))
    
    return points[point_IDs], aligned_points[point_IDs], point_IDs[0], label_id

def point_indices_from_group_160(points, aligned_points, seg_indices, group, label_mapping_dict):
    """
    Args:
        ...
        label_mapping_dict (dict): 由 get_raw2id_160_mapping 生成的字典 {str: int}
    """
    group_segments = np.array(group['segments'])
    raw_label = group['label']
    label_id = label_mapping_dict.get(raw_label, 0)
    point_IDs = np.where(np.isin(seg_indices, group_segments))
    
    return points[point_IDs], aligned_points[point_IDs], point_IDs[0], label_id



def loadOBJ(filePath):
    vertices = []
    uvs = []
    normals = []
    faceVertIDs = {}
    uvIDs = {}
    mtl = ""
    mtlfile = ""

    for line in open(filePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        
        if vals[0] == "mtllib":
            mtlfile = ' '.join(vals[1:])

        if vals[0] == "v":
            v = list(map(float, vals[1:4]))
            vertices.append(v)
            if len(vals) == 7:
                print("OBJ file have vertex colors")
                exit()

        if vals[0] == "vt":
            vt = list(map(float, vals[1:3]))
            uvs.append(vt)

        if vals[0] == "vn":
            vn = list(map(float, vals[1:4]))
            normals.append(vn)
        
        if vals[0] == "usemtl":
            mtl = vals[1]

        if vals[0] == "f":
            fvID = []
            uvID = []
            for f in vals[1:]:
                w = f.split("/")
                fvID.append(int(w[0]) - 1)
                uvID.append(int(w[1]) - 1)
            if mtl != "":
                if mtl in faceVertIDs:
                    faceVertIDs[mtl].append(fvID)
                else:
                    faceVertIDs[mtl] = []
                    faceVertIDs[mtl].append(fvID)

                if mtl in uvIDs:
                    uvIDs[mtl].append(uvID)
                else:
                    uvIDs[mtl] = []
                    uvIDs[mtl].append(uvID)
        
    vertices = np.array(vertices)
    uvs = np.array(uvs)
    normals = np.array(normals)

    return vertices, uvs, normals, faceVertIDs, uvIDs, mtlfile


def read_mtl_file(filePath):
    mtl_images = {}
    for line in open(filePath, "r"):
        vals = line.split()
        
        if len(vals) == 0:
            continue

        if vals[0] == "newmtl":
            newmtl = vals[1]

        if vals[0] == "map_Kd":
            image_name = ' '.join(vals[1:])
            if not newmtl in mtl_images:
                mtl_images[newmtl] = {}
            
            mtl_images[newmtl]["map_Kd"] = image_name

    return mtl_images



def texture_to_vertex_color(vt, uvIDs, mtl_images, dirpath):
    for mtl_name, uvs in uvIDs.items():
        if mtl_name in mtl_images:
            image_name = mtl_images[mtl_name]["map_Kd"]
            image_path = os.path.join(dirpath, image_name)
            ##v3　複数のtexはここを変更
            vc = uv_to_color(vt, Image.open(image_path))
            return vc
        else:
            print("Error Occured")
            exit()

    return 0


def concat_obj_data(vertices, vertex_color, vertex_normals):
    if len(vertex_normals) == 0:
        vertex_data = np.concatenate((vertices, vertex_color), axis=1)
    else:
        vertex_data = np.concatenate((vertices, vertex_color, vertex_normals), axis=1)
    vertex_data = vertex_data.astype(np.float32)
    return vertex_data


def uv_to_color(uv, image):

    if image is None or uv is None:
        return None

    uv = np.asanyarray(uv, dtype=np.float64)

    x = (uv[:, 0] * (image.width - 1))
    y = ((1 - uv[:, 1]) * (image.height - 1))

    x = x.round().astype(np.int64) % image.width
    y = y.round().astype(np.int64) % image.height

    colors = np.asanyarray(image.convert('RGBA'))[y, x]

    assert colors.ndim == 2 and colors.shape[1] == 4

    colors = colors[:, :3]
    return colors    


def convert_vertex_data_to_plyfile_format(vertex_data, with_vn):
    vertex_plyfile = [tuple(vd) for vd in vertex_data]
    if with_vn:
        vertex_plyfile_np = np.array(vertex_plyfile, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    else:
        vertex_plyfile_np = np.array(vertex_plyfile, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    return vertex_plyfile_np


def convert_face_data_to_plyfile_format(face_data):
    if len(face_data.keys()) == 1:
        face_plyfile = [(arr, ) for arr in list(face_data.values())[0]]
        face_plyfile_np = np.array(face_plyfile, dtype=[('vertex_indices', 'i4', (3,))])
        return face_plyfile_np
    return 0

class SAMPLE_METHODS(Enum):
    BBOX=1
    RADIUS=2

def find_neighbors(points, segments, search_method:SAMPLE_METHODS, receptive_field:float=0.50, selected_keys:list=None):
    '''
    Return dict {seg_idx : set(neighbor indices)}

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    segments : TYPE
        DESCRIPTION.
    search_method : SAMPLE_METHODS
        DESCRIPTION.
    receptive_field : float, optional
        DESCRIPTION. The default is 0.50.
    selected_keys : list, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # points = np.array(cloud.vertices.tolist())
    # segments = cloud.metadata['ply_raw']['vertex']['data']['objectId'].flatten()
    seg_ids = np.unique(segments)
    
    if selected_keys is not None:
        seg_ids = seg_ids.tolist()
        difference = set(selected_keys).difference(seg_ids)
        assert len(difference) == 0
        seg_ids = np.array(list(set(seg_ids).intersection(selected_keys)))
    
    ''' Get all segments '''
    trees = dict()
    segs_pts  = dict()
    bboxes = dict()
    # tmp_bboxes=dict()
    for idx in seg_ids:
        segs_pts[idx] = points[np.where(segments==idx)]
        trees[idx] = o3d.geometry.KDTreeFlann(segs_pts[idx][:,:3].transpose())
        bboxes[idx] = [segs_pts[idx][:,:3].min(0)-receptive_field,segs_pts[idx][:,:3].max(0)+receptive_field]
        # tmp_bboxes[idx] = [segs_pts[idx].min(0),segs_pts[idx].max(0)]
    
    ''' Building Box Method '''      
    segs_neighbors = dict()
    if search_method == SAMPLE_METHODS.BBOX:
        ''' Search neighbor of each segment '''
        for seg_idx in seg_ids:
            bbox_q = bboxes[seg_idx]
            seg_n = segs_neighbors[int(seg_idx)]=list()
            for seg_tar_idx in seg_ids:            
                if seg_idx == seg_tar_idx: continue
                bbox_t = bboxes[seg_tar_idx]
                if (bbox_q[0] > bbox_t[1]).sum() + (bbox_t[0] > bbox_q[1]).sum() > 0: 
                    continue                
                seg_n.append(int(seg_tar_idx))
    elif search_method == SAMPLE_METHODS.RADIUS:
        # search neighbor for each segments
        for seg_id in seg_ids:
            def f_nn(seg_id:int, trees:dict, bboxes:dict, segs:dict, radknn:float):
                pts = segs[seg_id]
                bbox_q = bboxes[seg_id]
                neighbors = set()
                for tree_idx, tree in trees.items():
                    if tree_idx == seg_id:continue
                    if tree_idx in neighbors: continue
                    bbox_t = bboxes[tree_idx]
                    if (bbox_q[0] > bbox_t[1]).sum() + (bbox_t[0] > bbox_q[1]).sum() > 0: 
                        continue
                    
                    for i in range(len(pts)):
                        point = pts[i]
                        k, _, _ = tree.search_radius_vector_3d(point,radknn)
                        if k > 0: 
                            neighbors.add(int(tree_idx))
                            break
                return neighbors
            neighbors = list(f_nn(seg_id, trees, bboxes, segs_pts, receptive_field))
            neighbors = [int(n) for n in neighbors]
            segs_neighbors[int(seg_id)] = neighbors
    return segs_neighbors

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def getLabelNames(path):
    import csv
    Scan3R=dict()
    NYU40=dict()
    Eigen13=dict()
    RIO27=dict()
    RIO7=dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            if not row[0].isnumeric():
                continue
            Scan3R[int(row[0])]=row[1]
            NYU40[int(row[2])] = row[3]
            Eigen13[int(row[4])] = row[5]
            RIO27[int(row[6])] = row[7]
            RIO7[int(row[8])] = row[9]
            # break
    return dict(sorted(Scan3R.items())),dict(sorted(NYU40.items())),\
           dict(sorted(Eigen13.items())),dict(sorted(RIO27.items())),dict(sorted(RIO7.items()))

def getLabelNameMapping(path):
    """
    return  toNameNYU40,toNameEigen,toNameRIO27,ttoNameRIO7
    """
    import csv
    raw=dict()
    toNameNYU40=dict()
    toNameEigen=dict()
    toNameRIO27=dict()
    toNameRIO7 =dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if not row[0].isnumeric():
                continue
            raw[row[1]]=row[1]
            toNameNYU40[row[1]] = row[3] if row[3] != '-' else 'none'
            toNameEigen[row[1]] = row[5] if row[5] != '-' else 'none'
            toNameRIO27[row[1]] = row[7] if row[7] != '-' else 'none'
            toNameRIO7[row[1]] = row[9] if row[9] != '-' else 'none'
    return raw, toNameNYU40,toNameEigen,toNameRIO27,toNameRIO7

def getLabelIdxMapping(path):
    """
    return  toNYU40,toEigen,toRIO27,toRIO7
    """
    import csv
    raw=dict()
    toNYU40=dict()
    toEigen=dict()
    toRIO27=dict()
    toRIO7 =dict()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            if not row[0].isnumeric():
                continue
            raw[int(row[0])] = int(row[0])
            toNYU40[int(row[0])] = int(row[2])
            toEigen[int(row[0])] = int(row[4])
            toRIO27[int(row[0])] = int(row[6])
            toRIO7[int(row[0])] = int(row[8])
            # break
    return raw,toNYU40,toEigen,toRIO27,toRIO7

def getLabelMapping(label_type:str,pth_mapping:str = ""):
    if pth_mapping == "":
        pth_mapping = os.path.join(
    'data/3rscan/meta_data', '3RScan.v2 Semantic Classes - Mapping.csv')
    pth_160 = os.path.join('data/3rscan/meta_data', 'classes160.txt')
    Scan3R528, NYU40,Eigen13,RIO27,RIO7 = getLabelNames(pth_mapping)
    NameScan3R528, toNameNYU40,toNameEigen13,toNameRIO27,toNameRIO7=getLabelNameMapping(pth_mapping)
    IdxScan3R528, toNYU40,toEigen13,toRIO27,toRIO7 = getLabelIdxMapping(pth_mapping)
    label_names=None
    label_name_mapping=None
    label_id_mapping=None
    label_type=label_type.lower()
    if label_type == 'nyu40':
        label_names = NYU40
        label_name_mapping = toNameNYU40
        label_id_mapping = toNYU40
    elif label_type == 'eigen13':
        label_names = Eigen13
        label_name_mapping = toNameEigen13
        label_id_mapping = toEigen13
    elif label_type == 'rio27':
        label_names = RIO27
        label_name_mapping = toNameRIO27
        label_id_mapping = toRIO27
    elif label_type == 'rio7':
        label_names = RIO7
        label_name_mapping = toNameRIO7
        label_id_mapping = toRIO7
    elif label_type == '3rscan':
        label_names=Scan3R528
        label_name_mapping=NameScan3R528
        label_id_mapping=IdxScan3R528
    elif label_type == '3rscan160':
        names = sorted(read_txt_to_list(pth_160))
        label_names = {k:v for k,v in enumerate(names,1)}
        n_to_id = {v:k for k,v in enumerate(names,1)}
        label_name_mapping = dict()
        label_id_mapping = dict()
        for k,v in NameScan3R528.items():
            label_name_mapping[k] = v if v in names else 'none'
        for k,v in Scan3R528.items():
            label_id_mapping[k] = 0 if v not in names else n_to_id[v]
        
    elif label_type == 'scannet20':
        label_names = NYU40
        label_name_mapping = toNameNYU40
        label_id_mapping = toNYU40
        
        label_names = {i+1:SCANNET20_Label_Names[i] for i in range(len(SCANNET20_Label_Names)) }
        for name, name40 in label_name_mapping.items():
            label_name_mapping[name] = name40 if name40 in label_names.values() else 'none'
        for id_f, id_40 in label_id_mapping.items():
            nyu40name = NYU40_Label_Names[id_40-1]
            if nyu40name in label_names.values():
                id_scan20 = list(label_names.values()).index(nyu40name)+1
                label_id_mapping[id_f] = id_scan20
            else:
                label_id_mapping[id_f] = 0
    else:
        raise RuntimeError('')
    return label_names, label_name_mapping, label_id_mapping

def get_label(ply_in, dataset_type, label_type):
    ply_raw = 'ply_raw' if 'ply_raw' in ply_in.metadata else '_ply_raw'
    data = ply_in.metadata[ply_raw]['vertex']['data']
    if dataset_type == None or dataset_type == '' or label_type == None:
        try:
            labels = data['objectId']
        except:
            labels = data['label']
    elif dataset_type == 'ScanNet':
        labels = data['label']
    elif dataset_type == '3RScan':
        if label_type == 'Segment':
            if type(data) is dict:
                labels = data['objectId'] if 'objectId' in data else data['label']
            else:
                labels = data['objectId'] if 'objectId' in data.dtype.names else data['label']
        elif label_type == 'NYU40':
            labels = data['NYU40']
        elif label_type == 'Eigen13':
            labels = data['Eigen13']
        elif label_type == 'RIO27':
            labels = data['RIO27']
        else:
            raise RuntimeError('unsupported label type:', label_type)
    else:
        raise RuntimeError('unsupported dataset type:', dataset_type)
    return labels