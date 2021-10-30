
import torch
import numpy as np
import argparse
import os

from model.gnn_model import GNN
from data_processing import graph_io,nifti_io, data_loader
import Filepaths
from data_processing.image_processing import uncrop_to_brats_size

from utils.hyperparam_helpers import EvalParamSet,DEFAULT_BACKGROUND_NODE_LOGITS,populate_hardcoded_hyperparameters
from model.networks import init_graph_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
Loads model weights from given file and generates predictions on graphs in specified data directory.
Data directory expected to created by preprocess script.
Outputs are saved in image form NOT in graph form. Output can either be logits (needed to train CNN) or predictions (for evaluation/downstream use)
Need to make sure that the type and shape of the weight file correspond to the model that the weights are being loaded into. See load_net_and_weights.
'''

#Make sure model type and hyperparameters correspond to the weight file.
def load_net_and_weights(model_type,weight_file):
    model_type="GSpool"
    gnn_hp = EvalParamSet(in_feats=20,out_classes=4,layer_sizes=[64]*6,gat_heads=None,gat_residuals=None)
    net = init_graph_net(model_type,gnn_hp)

    net.load_state_dict(torch.load(weight_file,map_location=device))
    net.eval()
    return net


#if preds is selected as output format, then they will additioanlly be expanded back to original BraTS size (240,240,155)
def save_predictions(net,dataset,save_format='logits'):
    global device
    for mri_id,graph,feats in dataset:
        graph.to(device)
        feats = torch.FloatTensor(feats).to(device)
        with torch.no_grad():
            logits = net(graph,feats)
        if(save_format=='preds'):
            save_voxel_preds(mri_id,dataset,logits)
        elif(save_format=='logits'):
            save_voxel_logits(mri_id,dataset,logits)
        else:
            raise ValueError(f"Unrecognized save format {save_format}")


def save_voxel_logits(mri_id,dataset,node_logits):
    global output_dir
    node_logits=node_logits.detach().cpu().numpy()
    supervoxel_partitioning = dataset.get_supervoxel_partitioning(mri_id)
    #add placeholder logits for healthy tissue
    node_logits = np.concatenate([node_logits,DEFAULT_BACKGROUND_NODE_LOGITS])
    voxel_logits = node_logits[supervoxel_partitioning]
    nifti_io.save_as_nifti(voxel_logits,f"{output_dir}{os.sep}{mri_id}_logits.nii.gz")

def save_voxel_preds(mri_id,dataset,node_logits):
    global output_dir
    _, predicted_nodes = torch.max(node_logits, dim=1)
    predicted_nodes=predicted_nodes.detach().cpu().numpy()
    supervoxel_partitioning = dataset.get_supervoxel_partitioning(mri_id)
    raw_data_crop = dataset.get_crop(mri_id)
    predicted_voxels = graph_io.project_nodes_to_img(supervoxel_partitioning,predicted_nodes)
    predicted_voxels = uncrop_to_brats_size(raw_data_crop,predicted_voxels)
    nifti_io.save_as_nifti(predicted_voxels,f"{output_dir}{os.sep}{mri_id}.nii.gz")


if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default=Filepaths.PROCESSED_DATA_DIR, help='path to the directory where data is stored',type=str)
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.',type=str)
    parser.add_argument('-o', '--output_dir', default=None,help='Directory to save predictions/logits to',type=str)
    parser.add_argument('-w', '--weight_file', default="", help='Path to a weights file to be loaded',type=str)
    parser.add_argument('-f', '--save_format', default="preds", help='preds or logits',type=str)

    args = parser.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Filepaths.GNN_LOGIT_DIR if args.save_format=="logits" else Filepaths.PRED_DIR
    dataset = data_loader.ImageGraphDataset(args.data_dir,args.data_prefix,read_image=False,read_graph=True,read_label=False)

    #hyperparams = populate_hardcoded_hyperparameters(args.model_type)
    net = load_net_and_weights(args.weight_file)
    save_predictions(net,dataset,args.save_format)
    print(f"Finished saving {args.save_format} generated by {args.weight_file} in folder {output_dir}")

