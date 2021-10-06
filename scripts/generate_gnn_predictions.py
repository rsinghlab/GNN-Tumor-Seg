
import torch
import numpy as np
import argparse
import os

from model.gnn_model import GNN
from utils.training_helpers import populate_hardcoded_hyperparameters
from data_processing import graph_io,nifti_io, data_loader
import Filepaths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Can output either logits or final predictions.


#Make sure hyperparams and weight file correspond
def load_model_and_weights(model_type,hp,weight_file):
    #weight_file = "/Users/camillo_stuff/singhlab/GNN-Tumor-Seg/temp_output/testRun1_f1.pt"

    #make model and return it. the net can be extracted later
    model = GNN(model_type,hp,None)
    model.net.load_state_dict(torch.load(weight_file,map_location=device))
    model.net.eval()
    return model


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
    node_logits = np.concatenate([node_logits,[[1.0,-1.0,-1.0,-1.0]]])
    voxel_logits = node_logits[supervoxel_partitioning]
    nifti_io.save_as_nifti(voxel_logits,f"{output_dir}{os.sep}{mri_id}_logits.nii.gz")

def save_voxel_preds(mri_id,dataset,node_logits):
    global output_dir
    _, predicted_nodes = torch.max(node_logits, dim=1)
    predicted_nodes=predicted_nodes.detach().cpu().numpy()
    supervoxel_partitioning = dataset.get_supervoxel_partitioning(mri_id)
    predicted_voxels = graph_io.project_nodes_to_img(supervoxel_partitioning,predicted_nodes)
    predicted_voxels = uncrop_to_brats_size(mri_id,dataset,predicted_voxels)
    nifti_io.save_as_nifti(predicted_voxels,f"{output_dir}{os.sep}{mri_id}.nii.gz")


#uncrops image back to full size (fills space around with healthy preds)
def uncrop_to_brats_size(mri_id,dataset,voxel_preds):
    raw_data_crop = dataset.get_crop(mri_id)
    #in the case that the size isnt standard you can also read in the original image again and extract the size from there, or just alter the preprocessor to also save the original size.
    brats_size_preds = np.zeros((240,240,155),dtype=np.int16)
    brats_size_preds[raw_data_crop]=voxel_preds
    return brats_size_preds


if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default=Filepaths.PROCESSED_DATA_DIR, help='path to the directory where data is stored',type=str)
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.',type=str)
    parser.add_argument('-o', '--output_dir', default=None,help='Directory to save predictions/logits to',type=str)
    parser.add_argument('-m', '--model_type', default="GSpool",help='What graph learning layer to use. GSpool, GSmean, GSgcn, GAT',type=str)
    parser.add_argument('-w', '--weight_file', default="", help='Path to a weights file to be loaded',type=str)
    parser.add_argument('-f', '--save_format', default="preds", help='preds or logits',type=str)

    args = parser.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Filepaths.GNN_LOGIT_DIR if args.save_format=="logits" else Filepaths.PRED_DIR
    dataset = data_loader.ImageGraphDataset(args.data_dir,args.data_prefix,read_image=False,read_graph=True,read_label=False)

    hyperparams = populate_hardcoded_hyperparameters(args.model_type)
    #can also evaluate model at this point by calling model.evaluate and setting dataset to read labels
    model = load_model_and_weights(args.model_type,hyperparams,args.weight_file)
    net = model.net
    save_predictions(net,dataset,args.save_format)
    print(f"Finished saving {args.save_format} generated by {args.weight_file} in folder {output_dir}")

