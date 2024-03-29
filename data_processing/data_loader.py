import torch
import os
import glob
import networkx as nx
import numpy as np
from dgl import from_networkx as to_dgl_graph
from dgl import batch as dgl_batch

from data_processing import nifti_io, graph_io
from data_processing.image_processing import determine_tumor_crop


'''
A Dataset similar to a torch dataset which iterates over all samples in a directory and returns the contents as numpy arrays.
Expects to receive a filepath to the output of the preprocess script which should have the following:
1.processed image (nifti)
2.label image (nifti)
3.networkx graph (json)
4.supervoxel partitioning (nifti)
5. (optionally) a .npy file containing the crop of the processed image relative to the original image


#Input#
dataset_root_dir: filepath to preprocessed dataset (generated by running preprocess script)
mri_start_string: a prefix that every image folder starts with (can be empty string)
read_image: whether to read in and return preprocessed images for each sample (only necessary for CNN model)
read_graph: whether to return graphs for each sample (for training GNN)
read_label: whether to read in labels. Will be returned in vector form (one label per node if )

#Output#

If graph:
Returns a DGL Graph, features for each node, and (optionally) labels for each node
If image:
Returns a numpy image array and (optionally) a numpy label array

'''
class ImageGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir,mri_start_string,read_image=True,read_graph=True,read_label=True):
        self.dataset_root_dir=dataset_root_dir
        self.all_ids = self.get_all_mris_in_dataset(dataset_root_dir,mri_start_string)
        self.read_image=read_image
        self.read_graph=read_graph
        self.read_label = read_label
        assert(self.read_graph or self.read_image)

    def get_all_mris_in_dataset(self,dataset_root_dir,mri_start_string):
        mri_folders = glob.glob(f"{dataset_root_dir}**/{mri_start_string}*/",recursive=True)
        mri_ids = [fp.split(os.sep)[-2] for fp in mri_folders]
        print(f"Found {len(mri_folders)} MRIs")
        return mri_ids

    def get_one(self,mri_id):
        if(self.read_graph and not self.read_image):
            return (mri_id, *self.get_graph(mri_id))
        elif(self.read_image  and not self.read_graph):
            return (mri_id, *self.get_image(mri_id))
        elif(self.read_image and self.read_graph):
            return (mri_id, *self.get_graph(mri_id), *self.get_image(mri_id))
        else:
            print("Invalid combination of flags")

    '''
    Reads in the saved networkx graph, converts it to a DGLGraph, normalizes the graph (not actually sure how useful this is),
    and returns the DGLGraph as well as a vector of node features and optionally labels.
    '''
    def get_graph(self,mri_id):
        nx_graph = graph_io.load_networkx_graph(f"{self.dataset_root_dir}{os.sep}{mri_id}{os.sep}{mri_id}_nxgraph.json")
        features = np.array([nx_graph.nodes[n]['features'] for n in nx_graph.nodes])
        if(self.read_label):
            labels = np.array([nx_graph.nodes[n]['label'] for n in nx_graph.nodes])
        G = to_dgl_graph(nx_graph)
        n_edges = G.number_of_edges()
        # normalization
        degs = G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        G.ndata['norm'] = norm.unsqueeze(1)
        #G.ndata['feat'] = features
        if(self.read_label):
            #G.ndata['label'] = labels
            return G, features, labels
        return G, features

    def get_voxel_labels(self,mri_id):
        fp=f"{self.dataset_root_dir}{os.sep}{mri_id}{os.sep}{mri_id}_label.nii.gz"
        return nifti_io.read_nifti(fp,np.int16)

    def get_image(self,mri_id):
        fp = f"{self.dataset_root_dir}{os.sep}{mri_id}{os.sep}{mri_id}_input.nii.gz"
        img = nifti_io.read_nifti(fp,np.float32)
        if(self.read_label):
            return img,self.get_voxel_labels(mri_id)
        else:
            return (img,)

    def get_supervoxel_partitioning(self,mri_id):
        fp=f"{self.dataset_root_dir}{os.sep}{mri_id}{os.sep}{mri_id}_supervoxels.nii.gz"
        return nifti_io.read_nifti(fp,np.int16)

    def get_crop(self,mri_id):
        fp=f"{self.dataset_root_dir}{os.sep}{mri_id}{os.sep}{mri_id}_crop.npy"
        return tuple(np.load(fp,allow_pickle=True))

    def __iter__(self):
        for mri_id in self.all_ids:
            yield self.get_one(mri_id)

    def __getitem__(self, index):
        mri_id = self.all_ids[index]
        return self.get_one(mri_id)

    def __len__(self):
        return len(self.all_ids)



'''
Responsible for reading and returning saved output logits upon request.
Upon instantiation, pass it a root directory where logits are saved.
These logits should have been created by running the generate_gnn_predictions script
with the save format specified as "logits".
When training the Refinement CNN, pass it this dataset and it will automatically load the logits
it requires to train. The first time it loads a set of logits it will also compute the closest crop
around the area predicted to be tumorous by the logits. This crop is then stored in a dictionary so
it does not have to be recomputed when the same logits are loaded during subsequent epochs.

Note that this dataset is not designed to be iterated over and is currently only capable
of reading in a file at a specified path.
'''
class PredLogitDataset():
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.mri_crops={}

    #crops GNN preds to only the part predicted to be tumorous
    def get_crop(self,mri_id):
        if(mri_id in self.mri_crops):
            return self.mri_crops[mri_id]
        else:
            preds = np.argmax(self.read_logits(mri_id),axis=-1)
            crop_idxs = determine_tumor_crop(preds)
            self.mri_crops[mri_id]=crop_idxs
            return crop_idxs

    def read_logits(self,mri_id):
        fp = f"{self.root_dir}{os.sep}{mri_id}_logits.nii.gz"
        try:
            logits = nifti_io.read_nifti(fp,np.float32)
        except FileNotFoundError:
            print("Couldnt open ",fp)
            raise FileNotFoundError(f"Logit file for {mri_id} not found in {self.root_dir}")
        return logits

    #can add some sort of normalization/standardization to logits
    def get_one(self,mri_id):
        logits=self.read_logits(mri_id)
        crop_idxs = self.get_crop(mri_id)
        return logits,crop_idxs



#Collation functions for torch dataloaders

def minibatch_graphs(samples):
    mri_ids,graphs,features, labels = map(list, zip(*samples))
    #print("Batch Mri Ids:",mri_ids)
    batched_graph = dgl_batch(graphs)
    return mri_ids,batched_graph, torch.FloatTensor(np.concatenate(features)), torch.LongTensor(np.concatenate(labels))


def collate_refinement_net(samples):
    mri,img,lab = samples[0]
    return mri,torch.FloatTensor(img),torch.LongTensor(lab)