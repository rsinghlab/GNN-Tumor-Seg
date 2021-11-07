
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv, GraphConv
from dgl.nn.pytorch.conv import SAGEConv

'''
Contains the actual neural network architectures.
Supports GraphSAGE with either the pool,mean,gcn, or lstm aggregator as well as GAT.
The input, output, and intermediate layer sizes can all be specified.
Typically will call init_graph_net and pass along the desired model and hyperparameters.

Also contains the CNN Refinement net which is a very simple 2 layer 3D convolutional neural network.
As input, it expects 8 channels, which are the concatenated 4 input modalities and 4 output logits of the GNN predictions.
'''


class GraphSage(nn.Module):
    def __init__(self,in_feats,layer_sizes,n_classes,aggregator_type,dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, layer_sizes[0], aggregator_type, feat_drop=dropout, activation=F.relu))
        # hidden layers
        for i in range(1,len(layer_sizes)):
            self.layers.append(SAGEConv(layer_sizes[i-1], layer_sizes[i], aggregator_type, feat_drop=dropout, activation=F.relu))
        # output layer
        self.layers.append(SAGEConv(layer_sizes[-1], n_classes, aggregator_type, feat_drop=0, activation=None))

    def forward(self,graph,features):
        h = features
        for layer in self.layers:
            h = layer(graph, h)
        return h


class GAT(nn.Module):
    def __init__(self,in_feats,layer_sizes,n_classes,heads,residuals,
                activation=F.elu,feat_drop=0,attn_drop=0,negative_slope=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.layers.append(GATConv(
            in_feats, layer_sizes[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for i in range(1, len(layer_sizes)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                layer_sizes[i-1] * heads[i-1], layer_sizes[i], heads[i],
                feat_drop, attn_drop, negative_slope, residuals[i], self.activation))
        # output projection
        self.layers.append(GATConv(
            layer_sizes[-1] * heads[-1], n_classes, 1,
            feat_drop, attn_drop, negative_slope, False, None))

    def forward(self,g, inputs):
        h = inputs
        for l in range(len(self.layers)-1):
            h = self.layers[l](g, h).flatten(1)
        # output projection
        logits = self.layers[-1](g, h).mean(1)
        return logits

def init_graph_net(model_type,hp):
    dropout = hp.feature_dropout if 'feature_dropout' in hp._fields else 0
    if(model_type=='GSpool'):
        net = GraphSage(in_feats=hp.in_feats,layer_sizes=hp.layer_sizes,n_classes=hp.out_classes,aggregator_type='pool',dropout=dropout)
    elif(model_type=='GSgcn'):
        net = GraphSage(in_feats=hp.in_feats,layer_sizes=hp.layer_sizes,n_classes=hp.out_classes,aggregator_type='gcn',dropout=dropout)
    elif(model_type=='GSmean'):
        net = GraphSage(in_feats=hp.in_feats,layer_sizes=hp.layer_sizes,n_classes=hp.out_classes,aggregator_type='mean',dropout=dropout)
    elif(model_type=='GAT'):
        net = GAT(in_feats=hp.in_feats,layer_sizes=hp.layer_sizes,n_classes=hp.out_classes,
                                heads=hp.gat_heads,residuals=hp.gat_residuals)
    else:
        raise Exception(f"Unknown model type: {model_type}")
    return net

class CnnRefinementNet(nn.Module):
    def __init__(self,in_feats,out_classes,layer_sizes):
        super().__init__()
        self.conv_layers=nn.ModuleList()
        self.conv_layers.append(nn.Conv3d(in_channels=in_feats,out_channels=layer_sizes[0],kernel_size=5,stride=1,padding=2,padding_mode='replicate'))
        self.conv_layers.append(nn.Conv3d(in_channels=layer_sizes[0],out_channels=out_classes,kernel_size=5,stride=1,padding=2,padding_mode='replicate'))
    
    def forward(self,comb_img_logits):
        h = F.relu(self.conv_layers[0](comb_img_logits))
        h = self.conv_layers[1](h)
        return h



