import torch
import torch.nn as nn
import itertools

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .encoder import SelfAttentionLayer, CrossAttentionLayer, FFN, PredBBox, _bbox_pred_to_bbox
from .position_embedding import PositionEmbeddingCoordsSine


class SelfAttentionLayer_pos(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, pos):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i in range(len(x)):
            z, _ = self.attn(x[i]+pos[i], x[i]+pos[i], x[i])
            z = self.dropout(z) + x[i]
            z = self.norm(z)
            out.append(z)
        return out
    
class VolumetricInteractionModule(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.context_proj = nn.Linear(d_model, d_model)

        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model), 
            nn.ReLU(),
            nn.Linear(d_model, 1), 
            nn.Sigmoid()
        )

    def forward(self, tgt_sub, tgt_obj, sub_centers, obj_centers, 
                scene_feats, scene_centers):
        geo_vec = sub_centers - obj_centers
        geo_emb = self.geo_mlp(geo_vec)
        anchor_points = (sub_centers + obj_centers) / 2.0

        dist = torch.cdist(anchor_points, scene_centers)
      
        mask_radius = 2.0 
        is_valid = (dist < mask_radius) 

        logits = -dist 

        logits = logits.masked_fill(~is_valid, -1e4) 
        
        attn_weights = nn.functional.softmax(logits, dim=-1)

        valid_mask = is_valid.any(dim=-1, keepdim=True).float()
        
        context_feats = torch.matmul(attn_weights, scene_feats)
        context_feats = self.context_proj(context_feats)

        gate_input = torch.cat([tgt_sub, tgt_obj, geo_emb], dim=-1)
        gate = self.gate_net(gate_input)

        return gate * context_feats * valid_mask, geo_emb
    
class AttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pos=None):
        """Forward pass.

        Args:
            q (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
            k (List[Tensor]): Keys of len batch_size,
                each of shape(n_queries_i, d_model).
            v (List[Tensor]): Values of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i in range(len(q)):
            if pos is not None:
                z, _ = self.attn(q[i]+pos[i], k[i]+pos[i], v[i])
            else:
                z, _ = self.attn(q[i], k[i], v[i])
            z = self.dropout(z) + q[i]
            z = self.norm(z)
            out.append(z)
        return out

@MODELS.register_module()
class GeometryAwareRelDecoder(BaseModule):
    """Encoder for the UniDet3D model.

        Args:
            num_layers (int): Number of transformer layers.
            datasets_classes (List[List[str]]): List of classes for 
                each dataset.
            in_channels (int): Number of input channels.
            d_model (int): Number of channels for model layers.
            num_heads (int): Number of head in attention layer.
            hidden_dim (int): Dimension of attention layer.
            dropout (float): Dropout rate for transformer layer.
            activation_fn (str): 'relu' of 'gelu'.
            datasets (List[str]): List of dataset names.
            angles (List[Bool]): Whether to use angle prediction.
        """

    def __init__(self, num_layers, datasets_classes, relation_classes, in_channels, 
                 d_model, num_heads, hidden_dim, dropout, activation_fn,
                 datasets, angles, localiztion_flag=False, vis_attn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.datasets = datasets
        self.angles = angles 
        self.d_model = d_model
        self.relation_classes = relation_classes
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.entity_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.triple_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, 2*d_model))
        self.cross_attn_sub_entity = nn.ModuleList([])
        self.cross_attn_sub_scene = nn.ModuleList([])
        self.sub_ffn_layers = nn.ModuleList([])
        self.cross_attn_obj_entity = nn.ModuleList([])
        self.sub_obj_self_attn_layers = nn.ModuleList([])
        self.cross_attn_obj_scene = nn.ModuleList([])
        self.obj_ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_sub_entity.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, return_attn=vis_attn))
            self.cross_attn_sub_scene.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, return_attn=vis_attn))
            self.sub_obj_self_attn_layers.append(
                AttentionLayer(d_model, num_heads, dropout))
            self.sub_ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
            self.cross_attn_obj_entity.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, return_attn=vis_attn))
            self.cross_attn_obj_scene.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, return_attn=vis_attn))
            self.obj_ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
            
        self.vim = VolumetricInteractionModule(d_model)
        self.rel_head = nn.Sequential(
            nn.Linear(d_model *4, d_model), nn.ReLU(),
            nn.Linear(d_model, len(self.relation_classes[0]))
        )
        self.entity_cross_attn_layers = nn.ModuleList([])
        self.entity_self_attn_layers = nn.ModuleList([])
        self.entity_ffn_layers = nn.ModuleList([])    
        for i in range(num_layers):
            self.entity_cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, return_attn=vis_attn))
            self.entity_self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.entity_ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        unique_cls = sorted(list(set(itertools.chain.from_iterable(
                            datasets_classes)))) + ['no_obj']
        self.outs_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.sub_outs_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.obj_outs_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.localiztion_flag = localiztion_flag
        if self.localiztion_flag:
            self.out_localization = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 1))
        self.datasets_cls_idxs = []
        for dataset_classes in datasets_classes:
            dataset_cls_idxs = []
            for cls in dataset_classes:
                dataset_cls_idxs.append(unique_cls.index(cls))
            self.datasets_cls_idxs.append(dataset_cls_idxs + [-1])
        self.sub_out_norm = nn.LayerNorm(d_model)
        self.obj_out_norm = nn.LayerNorm(d_model)
        self.out_bboxes = PredBBox(d_model, 8)
        self.sub_bboxes = PredBBox(d_model, 8)
        self.obj_bboxes = PredBBox(d_model, 8)
        self.sub_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.obj_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.pos_emb=PositionEmbeddingCoordsSine(normalize=False, pos_type='sine', d_pos=d_model)


    def _forward_head(self, feats, scene_feats, tgt_subs, tgt_objs, sp_centers, sp_centers_rel, scene_centers, datasets_names):
        """Prediction head forward.

        Args:
            feats (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, d_model).
            sp_centers (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, 3) representing 
                the spatial centers for the predicted 
                bounding boxes.
            datasets_names (List[str]): A list of dataset names 
                corresponding to each input feature tensor.

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                    - List[Tensor]: Classification predictions 
                        of length `batch_size`, with each tensor 
                        having a shape of (n_bboxes_i, n_classes + 1).
                    - List[Tensor]: Bounding box predictions of 
                        length `batch_size`, with each tensor having 
                        a shape of (n_bboxes_i, 7) (or 6 if `angles` 
                        are not applicable).
        """
        cls_preds, pred_bboxes, localization_preds = [], [], []
        sub_cls_preds, sub_pred_bboxes, obj_cls_preds, obj_pred_bboxes, rel_preds = [], [], [], [], []
        for i in range(len(feats)):
            norm_query = self.out_norm(feats[i])
            tgt_sub = self.sub_out_norm(tgt_subs[i])
            tgt_obj = self.obj_out_norm(tgt_objs[i])
            idx = self.datasets.index(datasets_names[i])
            class_idxs = norm_query.new_tensor(
                            self.datasets_cls_idxs[idx]).long()
            sub_cls_preds.append(self.sub_outs_cls(tgt_sub)[:, class_idxs])
            obj_cls_preds.append(self.obj_outs_cls(tgt_obj)[:, class_idxs])
            cls_preds.append(self.outs_cls(norm_query)[:, class_idxs])
            pred_bbox = self.out_bboxes(norm_query)
            sub_pred_bbox = self.out_bboxes(tgt_sub)
            obj_pred_bbox = self.out_bboxes(tgt_obj)
            if not self.angles[idx]:
                pred_bbox = pred_bbox[:, :6]
                sub_pred_bbox = sub_pred_bbox[:, :6]
                obj_pred_bbox = obj_pred_bbox[:, :6]
            localization_pred = self.out_localization(norm_query) if self.localiztion_flag else None
            localization_preds.append(localization_pred)
            pred_bbox = _bbox_pred_to_bbox(sp_centers[i], pred_bbox)
            sub_pred_bbox = _bbox_pred_to_bbox(sp_centers_rel[i], sub_pred_bbox)
            obj_pred_bbox = _bbox_pred_to_bbox(sp_centers_rel[i], obj_pred_bbox)
            curr_sub_center = sub_pred_bbox[:, :3]
            curr_obj_center = obj_pred_bbox[:, :3]

            context_feat, geo_emb = self.vim(
                tgt_sub, tgt_obj, 
                curr_sub_center, curr_obj_center,
                scene_feats[i], scene_centers[i]
            )
            rel_pred = self.rel_head(torch.cat([tgt_sub, tgt_obj, geo_emb, context_feat], dim=-1))
            rel_preds.append(rel_pred)
            pred_bboxes.append(pred_bbox)
            sub_pred_bboxes.append(sub_pred_bbox)
            obj_pred_bboxes.append(obj_pred_bbox)

        return cls_preds, pred_bboxes, localization_preds, sub_cls_preds, sub_pred_bboxes, obj_cls_preds, obj_pred_bboxes, rel_preds
    
    

    def forward(self, queries, queries_rel, x, sp_centers, sp_centers_rel, scene_centers, datasets_names):
            """Forward pass.
            
            Args:
                x (List[Tensor]): of len batch_size, each of shape
                    (n_bboxes_i, in_channels).
                sp_centers (List[Tensor]): A list of spatial centers 
                    for the superpoints, with a length of `batch_size`. 
                    Each tensor has a shape of (n_bboxes_i, 3).
                datasets_names (List[str]): A list of dataset names 
                    corresponding to each input feature tensor.
            Returns:
                Dict: with cls_preds, pred_bboxes and aux_outputs.
            """
            cls_preds, pred_bboxes, localization_preds, rel_preds, sub_cls_preds, sub_pred_bboxes, obj_cls_preds, obj_pred_bboxes = [], [], [], [], [], [], [], []
            scenes_feats = [self.input_proj(y) for y in x]
            entity_queries = [self.entity_proj(y) for y in queries]
            triple_queries = [self.triple_proj(y) for y in queries_rel]
            tgt_sub, tgt_obj = self.triple_to_sub_obj(triple_queries, self.d_model)
            cls_pred, pred_bbox, localization_pred, sub_cls_pred, sub_pred_bbox, obj_cls_pred, obj_pred_bbox, rel_pred = \
                self._forward_head(entity_queries, scenes_feats, tgt_sub, tgt_obj, sp_centers, sp_centers_rel, scene_centers, datasets_names)

            sub_cls_preds.append(sub_cls_pred)
            sub_pred_bboxes.append(sub_pred_bbox)
            obj_cls_preds.append(obj_cls_pred)
            obj_pred_bboxes.append(obj_pred_bbox)
            cls_preds.append(cls_pred)
            pred_bboxes.append(pred_bbox)
            localization_preds.append(localization_pred)
            rel_preds.append(rel_pred)
            for i in range(self.num_layers):
                entity_queries = self.entity_self_attn_layers[i](entity_queries)
                entity_queries = self.entity_cross_attn_layers[i](entity_queries, scenes_feats)
                entity_queries = self.entity_ffn_layers[i](entity_queries)

                tgt_so = [torch.cat([s, o], dim=0) for s, o in zip(tgt_sub, tgt_obj)]
                so_queries = self.sub_obj_self_attn_layers[i](tgt_so, tgt_so, tgt_so)
                tgt_sub, tgt_obj = self.so_to_sub_obj(so_queries)
                tgt_sub = self.cross_attn_sub_scene[i](tgt_sub, scenes_feats)
                tgt_sub = self.cross_attn_sub_entity[i](tgt_sub, entity_queries)
                tgt_sub = self.sub_ffn_layers[i](tgt_sub)

                tgt_obj = self.cross_attn_obj_scene[i](tgt_obj, scenes_feats)
                tgt_obj = self.cross_attn_obj_entity[i](tgt_obj, entity_queries)
                tgt_obj = self.obj_ffn_layers[i](tgt_obj)

                cls_pred, pred_bbox, localization_pred, sub_cls_pred, sub_pred_bbox, obj_cls_pred, obj_pred_bbox, rel_pred = \
                        self._forward_head(entity_queries, scenes_feats, tgt_sub, tgt_obj, sp_centers, sp_centers_rel, scene_centers, datasets_names)
                sub_cls_preds.append(sub_cls_pred)
                sub_pred_bboxes.append(sub_pred_bbox)
                obj_cls_preds.append(obj_cls_pred)
                obj_pred_bboxes.append(obj_pred_bbox)
                cls_preds.append(cls_pred)
                pred_bboxes.append(pred_bbox)
                localization_preds.append(localization_pred)
                rel_preds.append(rel_pred)
            aux_outputs = [
                dict(
                    cls_preds=cls_pred,
                    bboxes=bboxes,
                    localizations=localization_pred,
                    rel_preds=rel_pred,
                    obj_cls_preds=obj_cls_pred,
                    obj_pred_bboxes=obj_pred_bbox,
                    sub_cls_preds=sub_cls_pred,
                    sub_pred_bboxes=sub_pred_bbox,
                    )
                for cls_pred, bboxes, localization_pred, rel_pred, obj_cls_pred, obj_pred_bbox, sub_cls_pred, sub_pred_bbox in zip(
                    cls_preds[:-1], pred_bboxes[:-1], localization_preds[:-1], rel_preds[:-1], obj_cls_preds[:-1], obj_pred_bboxes[:-1], sub_cls_preds[:-1], sub_pred_bboxes[:-1])]
            return dict(
                cls_preds=cls_preds[-1],
                bboxes=pred_bboxes[-1],
                localizations=localization_preds[-1],
                rel_preds=rel_preds[-1],
                obj_cls_preds=obj_cls_preds[-1],
                obj_pred_bboxes=obj_pred_bboxes[-1],
                sub_cls_preds=sub_cls_preds[-1],
                sub_pred_bboxes=sub_pred_bboxes[-1],
                aux_outputs=aux_outputs)

    def with_pos_embed(self, tensor, pos):
        queries = []
        for i in range(len(tensor)):
            if pos is None:
                queries.append(tensor[i])
            else:
                queries.append(tensor[i] + pos[i])
        return queries

    def triple_to_sub_obj(self, triple_queries, d_model):
        sub_queries, obj_queries = [], []
        for i in range(len(triple_queries)):
            tgt_sub, tgt_obj = torch.split(triple_queries[i], d_model, dim=-1)
            sub_queries.append(tgt_sub)
            obj_queries.append(tgt_obj)
        return sub_queries, obj_queries

    def so_to_sub_obj(self, tgt_so):
        sub_queries, obj_queries = [], []
        for i in range(len(tgt_so)):
            tgt_sub, tgt_obj = torch.split(tgt_so[i], tgt_so[i].shape[0] // 2, dim=0)
            sub_queries.append(tgt_sub)
            obj_queries.append(tgt_obj)
        return sub_queries, obj_queries
