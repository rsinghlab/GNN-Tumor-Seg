from unicodedata import decimal
import torch
import numpy as np
import argparse
import os

from model.networks import CnnRefinementNet,init_graph_net
from data_processing import graph_io,nifti_io, data_loader
from data_processing.image_processing import determine_tumor_crop,uncrop_to_brats_size
from .preprocess_dataset import swap_labels_to_brats
import Filepaths
from utils.hyperparam_helpers import EvalParamSet,DEFAULT_BACKGROUND_NODE_LOGITS
from model.cnn_model import combine_logits_and_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#takes filepaths to both GNN and CNN weight files and generates predictions by first predicting with GNN and then refining with CNN.

'''
Once both a GNN and CNN have been trained using the provided scripts, can generate predictions by chaining the two models.
Provide the type of GNN and the paths to the weight files and ensure that the hyperparameters below match the expected shape
of each model's layers and sizes.
Then just specify the directory of images to be predicted on. These images should be preprocessed using the provided script.
'''


#Make sure hyperparams and weight file correspond
def load_nets(gnn_type,gnn_weights,cnn_weights):
    global device
    gnn_hp = EvalParamSet(in_feats=20,out_classes=4,layer_sizes=[256]*4,gat_heads=None,gat_residuals=None)
    cnn_hp = EvalParamSet(in_feats=8,out_classes=4,layer_sizes=[16],gat_heads=None,gat_residuals=None)
    graph_net = init_graph_net(gnn_type,gnn_hp)
    conv_net = CnnRefinementNet(cnn_hp.in_feats,cnn_hp.out_classes,cnn_hp.layer_sizes)
    graph_net = graph_net.to(device)
    conv_net = conv_net.to(device)
    graph_net.load_state_dict(torch.load(gnn_weights,map_location=device))
    conv_net.load_state_dict(torch.load(cnn_weights,map_location=device))
    graph_net.eval()
    conv_net.eval()
    return graph_net, conv_net


def save_predictions(graph_net,conv_net,dataset):
    global output_dir
    for mri,graph,node_feats,img in dataset:
        try:
            supervoxel_partitioning = dataset.get_supervoxel_partitioning(mri)
            raw_data_crop = dataset.get_crop(mri)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Couldnt predict {mri} because couldn't read in a required file: {e.message}")
        with torch.no_grad():
            pred = predict_one_sample(graph_net,conv_net,graph,node_feats,img,supervoxel_partitioning)
        pred = uncrop_to_brats_size(raw_data_crop,pred)
        pred = swap_labels_to_brats(pred)
        nifti_io.save_as_nifti(pred,f"{output_dir}{os.sep}{mri}.nii.gz")


def predict_one_sample(graph_net,conv_net,graph,node_feats,img,supervoxel_partitioning):
    global device
    graph = graph.to(device)
    node_feats = torch.FloatTensor(node_feats).to(device)
    img = torch.FloatTensor(img).to(device)
    node_logits = graph_net(graph,node_feats)
    node_logits = torch.cat([node_logits,torch.FloatTensor(DEFAULT_BACKGROUND_NODE_LOGITS).to(device)],axis=0)
    voxel_logits = node_logits[supervoxel_partitioning]
    tumor_crop_idxs = determine_tumor_crop(voxel_logits.detach().cpu().numpy().argmax(axis=-1))
    cnn_in = combine_logits_and_image(voxel_logits,img,tumor_crop_idxs).to(device)
    refined_voxel_logits = conv_net(cnn_in)
    cropped_volume_preds = torch.argmax(refined_voxel_logits.squeeze(0),axis=0).detach().cpu().numpy()
    brain_volume_preds = np.zeros_like(supervoxel_partitioning,dtype=np.int16)
    brain_volume_preds[tumor_crop_idxs]=cropped_volume_preds
    return brain_volume_preds





if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default=Filepaths.PROCESSED_DATA_DIR, help='path to the directory where data is stored',type=str)
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.',type=str)
    parser.add_argument('-o', '--output_dir', default=None,help='Directory to save predictions to',type=str)
    parser.add_argument('-m', '--gnn_type', default="GSpool",help='What graph learning layer the saved model uses. GSpool, GSmean, GSgcn, GAT',type=str)
    parser.add_argument('-c', '--cnn_weights', default="", help='Path to weights file for convolutional net',type=str)
    parser.add_argument('-g', '--gnn_weights', default="", help='Path to weights file for graph net',type=str)

    args = parser.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Filepaths.GNN_LOGIT_DIR if args.save_format=="logits" else Filepaths.PRED_DIR
    output_dir = os.path.expanduser(output_dir)
    if not os.path.isdir(output_dir):
        print(f"Creating save directory: {output_dir}")
        os.makedirs(output_dir)
    dataset = data_loader.ImageGraphDataset(os.path.expanduser(args.data_dir),args.data_prefix,read_image=True,read_graph=True,read_label=False)

    graph_net,conv_net = load_nets(args.gnn_type,os.path.expanduser(args.gnn_weights),os.path.expanduser(args.cnn_weights))
    save_predictions(graph_net,conv_net,dataset)
    print(f"Finished saving predictions generated by {args.gnn_weights} and {args.cnn_weights} in folder {output_dir}")