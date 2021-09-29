
import torch
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import networkx as nx
import os
import networkx as nx
try:
    from dgl import DGLGraph
    from dgl import batch as dgl_batch
except:
    #Print an error but basically ignore it- this happens when i have dgl cuda version installed but am running cpu code
    print("Ignoring dgl imports. Assuming you only want MRIProcessor.")

from . import graphio

from .common_functions import determine_tumor_crop

#temp
import matplotlib.pyplot as plt


#If label_ext is None, then dont read in or return labels
#Accepts a list of modality extensions. Assumes each patient sample has exactly one image with each of the extensions.
class MRIProcessor(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir,mri_start_string,removed_mris,modality_extensions,label_ext,standardization_stats=None):
        self.all_scans,self.scan_to_fps = self.get_all_mris_in_dataset(dataset_root_dir,mri_start_string,removed_mris)
        self.modality_extensions=modality_extensions
        self.num_modalities = len(modality_extensions)
        self.label_ext = label_ext
        data_stats=self.compute_dataset_stats() if standardization_stats==None else standardization_stats
        self.dataset_mean = data_stats[0]
        self.dataset_std = data_stats[1]
        
    def get_all_mris_in_dataset(self,dataset_root_dir,mri_start_string,removed_mris):
        mri_folders = glob.glob(f"{dataset_root_dir}**/{mri_start_string}*/",recursive=True)
        mri_folders = self.remove_incomplete_mris(mri_folders,removed_mris)
        scan_dic = {fp.split("/")[-2]+"/":fp for fp in mri_folders}
        #print(mri_folders)
        print(f"Found {len(mri_folders)} MRIs")
        return mri_folders,scan_dic

    def remove_incomplete_mris(self,mri_folders,removed_mri_ids):
        return [fp for fp in mri_folders if fp.split("/")[-2] not in removed_mri_ids ]

    def get_specific_mri(self,mri_scan):
        full_path = self.scan_to_fps[mri_scan]
        return self.process_one_mri(full_path)


    def process_one_mri(self, scan_full_path):
        image_data = self.read_in_patient_sample(scan_full_path)
        crop_indices = get_img_crop(image_data)
        cropped_data = image_data[crop_indices]
        if(self.label_ext):
            label_array = self.read_in_labels(scan_full_path)
            cropped_labels = label_array[crop_indices]
        else:
            cropped_labels=None
        scan_id = scan_full_path.split(os.sep)[-2]+"/"

        #cropped_data, cropped_labels = pad_to_even(cropped_data,cropped_labels)

        #standardized_data = standardize_img(cropped_data,self.dataset_mean,self.dataset_std)
        #normalized_data = normalize_img(standardized_data)

        normalized_data = normalize_img(cropped_data)
        standardized_data = standardize_img(normalized_data,self.dataset_mean,self.dataset_std)

        #return scan_id,standardized_data[:,:,80],cropped_labels[:,:,80]
        return scan_id,standardized_data,cropped_labels

    def __iter__(self):
        for mri_path in self.all_scans:
            yield self.process_one_mri(mri_path)


    
    def __getitem__(self, index):
        full_path = self.all_scans[index]
        return self.process_one_mri(full_path)

    #returns nxmxrxC np array where C is number of modalities requested/available
    #or nxmxr if only one modality
    #does NOT do normalization
    def read_in_patient_sample(self,scan_full_path):
        modality_imgs = []
        for root, _, files in os.walk(scan_full_path):
            for ext in self.modality_extensions:
                for filename in files:
                    if filename.endswith(ext):
                        filepath = os.path.join(root, filename)
                        mod_img = nib.load(filepath)
                        img_data = mod_img.get_fdata()
                        modality_imgs.append(img_data)
        #check that all the modalities were present in the folder
        assert(len(modality_imgs)==self.num_modalities)

        patient_sample = np.stack(modality_imgs,3) if self.num_modalities>1 else modality_imgs[0]
        #scan_id = scan_full_path.split("/")[-2]
        #print(f"Finished reading in and normalizing MRI {scan_id}")
        return patient_sample


    def read_in_labels(self,scan_full_path):
        for filename in os.listdir(scan_full_path):
            if filename.endswith(self.label_ext):
                label_img = nib.load(scan_full_path+os.sep+filename)
                return label_img.get_data()
        raise FileNotFoundError(f"Label image not found in folder: {scan_full_path}")

    #returns the mean and standard deviation of all MRIs
    def compute_dataset_stats(self):
        print("Computing dataset mean and SD")
        img_means=[]
        img_deviations=[]
        for mri_path in self.all_scans:
            img = self.read_in_patient_sample(mri_path)
            #lab = self.read_in_labels(mri_path)
            if(len(img.shape)>3):
                #mask = np.logical_and(img[:,:,:,0]>0.001,lab==0)
                img = img[img[:,:,:,0].nonzero()]
                #img=img[mask]
                img = img/np.quantile(img,0.995,axis=0)
                mu = np.mean(img,axis=0)
                sigma = np.std(img,axis=0)
            else:
                mu = np.mean(img)
                sigma = np.std(img)
            img_means.append(mu)
            img_deviations.append(sigma)
        dataset_mean = np.median(img_means,axis=0)
        dataset_deviation = np.median(img_deviations,axis=0)
        print(f"Mean:{dataset_mean}, SD: {dataset_deviation} ")
        return dataset_mean,dataset_deviation



    def __len__(self):
        return len(self.all_scans)



class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir,mri_start_string,include_labels=True):
        self.saved_graphs_folder = dataset_root_dir
        self.include_labels=include_labels
        self.mri_ids, self.mri_data= self.get_all_graphs_in_dataset(mri_start_string)
        print("Finished reading all graphs into memory: ")
        print(self.mri_ids)


        

    def get_all_graphs_in_dataset(self,mri_start_string):
        mri_data = []
        mri_ids = [ f.name+"/" for f in os.scandir(self.saved_graphs_folder) if (f.is_dir() and mri_start_string in f.name) ]
        for mri in mri_ids:
            out = self.get_specific_mri(mri)
            obj = (mri,*out)
            mri_data.append(obj)

        return mri_ids, mri_data

    #This is inefficent: Do not call in a loop! (iterate over list instead)
    def get_specific_mri(self,mri_id):
        full_path = self.saved_graphs_folder+mri_id
        return self.pre_process_graph(full_path,False)


    def __getitem__(self, index):
        if(type(index) is list):
            return [self.mri_data[i] for i in index]
        return self.mri_data[index]

    def pre_process_graph(self,full_path_to_scan,adjust_self_loops=True):
        # graph preprocess and calculate normalization factor
        G = graphio.load_networkx_graph(full_path_to_scan)
        features = np.array([G.nodes[n]['features'] for n in G.nodes])
        if(self.include_labels):
            labels = np.array([G.nodes[n]['label'] for n in G.nodes])
        # add self loop
        if adjust_self_loops:
            G.remove_edges_from(nx.selfloop_edges(G))
            G.add_edges_from(zip(G.nodes(), G.nodes()))
        g = DGLGraph(G)
        n_edges = g.number_of_edges()
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        g.ndata['feat'] = features
        if(self.include_labels):
            g.ndata['label'] = labels
            return g, features, labels
        return g, features

    def __len__(self):
        return len(self.mri_ids)


class PreprocessedDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir):
        self.dataset_root_dir = dataset_root_dir
        self.all_scans = self.get_all_mris_in_dataset(dataset_root_dir)
        
    def get_all_mris_in_dataset(self,dataset_root_dir):
        patient_samples = glob.glob(f"{dataset_root_dir}/*.npz",recursive=False)
        print(f"Found {len(patient_samples)} MRIs")
        return patient_samples

    def load_one_mri(self,path):

        with np.load(path) as data:
            img = data['input'].astype('float32')
            lab = data['label'].astype('int64')
        scan_id = path.split(os.sep)[-1][:-4]
        return scan_id, img, lab

    #hasnt been tested
    def get_specific_mri(self,scan_id):
        scan_path = f"{self.dataset_root_dir}{scan_id}.npz"
        with np.load(scan_path) as data:
            img = data['input'].astype('float32')
            lab = data['label'].astype('int64')
        return scan_id, img, lab


    def __iter__(self):
        for mri_path in self.all_scans:
            try:
                yield self.load_one_mri(mri_path)
            except Exception as e:
                print(e)
                print("Couldn't open file", mri_path)
                yield None, None, None


    def __getitem__(self, index):
        full_path = self.all_scans[index]
        try:
            return self.load_one_mri(full_path)
        except Exception as e:
            print("Couldnt open file",full_path)
            return self.__getitem__(index+1)

    def __len__(self):
        return len(self.all_scans)



# class CombinedGraphImgDataset(torch.utils.data.Dataset):
#     def __init__(self, graphs_root_dir,imgs_root_dir,mri_prefix):
#         self.graphs_root_dir = graphs_root_dir
#         self.imgs_root_dir = imgs_root_dir
#         self.all_scans = self.get_all_mris_in_dataset(mri_prefix)
        
#     def get_all_mris_in_dataset(self,mri_prefix):
#         mri_ids = [ f.name for f in os.scandir(self.graphs_root_dir) if (f.is_dir() and mri_prefix in f.name) ]
#         print(f"Found {len(mri_ids)} MRIs")
#         return mri_ids

#     def load_one_mri(self,path):
#         with np.load(path) as data:
#             img = data['input'].astype('float32')
#             lab = data['label'].astype('int64')
#         return img, lab

#     def load_graph_and_clustering(self,path):
#         G = graphio.load_networkx_graph(path)
#         features = np.array([G.nodes[n]['features'] for n in G.nodes])
#         g = DGLGraph(G)
#         degs = g.in_degrees().float()
#         norm = torch.pow(degs, -0.5)
#         norm[torch.isinf(norm)] = 0
#         g.ndata['norm'] = norm.unsqueeze(1)
#         g.ndata['feat'] = features
#         #clustering===region_img
#         clustering = np.load(path+"clustering.npy")
#         return g, features, clustering


#     def __getitem__(self, index):
#         mri_id = self.all_scans[index]
#         img_path=f"{self.imgs_root_dir}{mri_id}.npz"
#         graph_path=f"{self.graphs_root_dir}{mri_id}/"
#         try:
#             img,lab =  self.load_one_mri(img_path)
#         except Exception as e:
#             print("Couldnt open file",mri_id)
#             print(e)
#             return self.__getitem__(index+1)
#         g,feats,region_img=self.load_graph_and_clustering(graph_path)
#         return g,feats,img,region_img,lab

#     def __len__(self):
#         return len(self.all_scans)


class CnnRefinementDataset(torch.utils.data.Dataset):
    def __init__(self, graph_logit_dir,imgs_root_dir,mri_prefix,read_labels=True):
        self.graph_logit_dir = graph_logit_dir
        self.imgs_root_dir = imgs_root_dir
        self.all_scans = self.get_all_mris_in_dataset(mri_prefix)
        self.img_shapes,self.tumor_crops = self.get_tumor_crop_for_all_mris()
        self.read_labels = read_labels
        
    def get_all_mris_in_dataset(self,mri_prefix):
        mri_ids = [ f.name[:-4] for f in os.scandir(self.graph_logit_dir) if (mri_prefix in f.name) ]
        print(f"Found {len(mri_ids)} MRIs")
        return mri_ids

    def get_tumor_crop_for_all_mris(self):
        crops = {}
        img_shapes = {}
        for mri_id in self.all_scans:
            fp = f"{self.graph_logit_dir}{mri_id}.npy"
            preds = np.argmax(self.load_logits(fp),axis=-1)
            crop_idxs = determine_tumor_crop(preds)
            crops[mri_id]=crop_idxs
            img_shapes[mri_id]=preds.shape
        return img_shapes,crops


    def load_one_mri(self,path):
        with np.load(path) as data:
            img = data['input'].astype('float32')
            lab = data['label'].astype('int64') if self.read_labels else None
        return img, lab

    def load_logits(self,path):
        logits = np.load(path).astype('float32')
        return logits

    def __getitem__(self, index):
        mri_id = self.all_scans[index]
        img_path=f"{self.imgs_root_dir}{mri_id}.npz"
        logits_path=f"{self.graph_logit_dir}{mri_id}.npy"
        try:
            img,lab =  self.load_one_mri(img_path)
        except Exception as e:
            print("Couldnt open file",mri_id)
            return mri_id,None,None,None
        logits=self.load_logits(logits_path)
        c = self.tumor_crops[mri_id]
        #crop them both to correct size and return
        if(self.read_labels):
            return mri_id,img[c],logits[c],lab[c]
        else:
            return mri_id,img[c],logits[c],None

    def __len__(self):
        return len(self.all_scans)



def minibatch_graphs(samples):
    mri_ids,graphs,features, labels = map(list, zip(*samples))
    #print("Batch Mri Ids:",mri_ids)
    batched_graph = dgl_batch(graphs)
    return mri_ids,batched_graph, torch.FloatTensor(np.concatenate(features)), torch.LongTensor(np.concatenate(labels))


def minibatch_combined(samples):
    g,feats,img,region_img,lab = samples[0]
    return g,torch.FloatTensor(feats),torch.FloatTensor(img).movedim(-1,0).unsqueeze(0),torch.LongTensor(region_img), torch.LongTensor(lab).unsqueeze(0)




