
import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing

def reset_parameters_with_activation(mm,nonlinearity):
    def process(m,nonlinearity):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),nonlinearity=nonlinearity)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),nonlinearity=nonlinearity)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
        else:
            print('unknown input type', m.__name__)
    
    if isinstance(mm , list):
        for m in mm: 
            process(m,nonlinearity)
    else:
        process(mm,nonlinearity)

def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(torch.nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or on_last:
            reset_parameters_with_activation(
                layers[-1], nonlinearity=activation)
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*layers)

class TripletGCN(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='mean', with_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.nn1 = build_mlp([dim_node*2+dim_edge, dim_hidden, dim_hidden*2+dim_edge],
                             do_bn=with_bn, on_last=True)
        self.nn2 = build_mlp([dim_hidden, dim_hidden, dim_node], do_bn=with_bn)

        self.reset_parameter()

    def reset_parameter(self):
        pass

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(
            edge_index, x=x, edge_feature=edge_feature)
        gcn_x = x + self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)  # .view(b,-1)
        new_x_i = x[:, :self.dim_hidden]
        new_e = x[:, self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:, (self.dim_hidden+self.dim_edge):]
        x = new_x_i+new_x_j
        return [x, new_e]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class TripletGCNModel(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature