import numpy as np
import glob
import os
import concurrent.futures
import argparse

from data_processing import nifti_io, graph_io
from data_processing.image_processing import *
from mri2graph.graphgen import img2graph
import Filepaths


#combines preprocessing dataset, swapping labels and building graph

LABEL_MAP={4:3,2:1,1:2}
STANDARDIZATION_STATS=([0.4645,0.6625,0.4064,0.3648],[0.1593,0.1703,0.1216,0.1627])
N_THREADS = 3



class DataPreprocessor():
    def __init__(self,args):
        print("Preparing to build graphs with: ",args)
        #Graph specs
        self.num_nodes = args.num_nodes
        self.num_neighbors = args.num_neighbors if args.num_neighbors!=0 else None
        self.boxiness_coef = args.boxiness
        #Data specs
        self.data_dir=args.data_dir if args.data_dir else Filepaths.INPUT_MRI_DIR
        self.output_dir = args.output_dir if args.output_dir else f"{Filepaths.PROCESSED_DATA_DIR}_{args.num_nodes}_{args.boxiness}_{args.num_neighbors}"
        self.mri_prefix = args.data_prefix
        self.modality_extensions=args.modality_extensions
        self.label_extension = args.label_extension
        self.include_labels = args.label_extension is not None

        self.all_ids,self.id_to_fp = self.get_all_mris_in_dataset()
        data_stats=self.compute_dataset_stats() if STANDARDIZATION_STATS is None else STANDARDIZATION_STATS
        self.dataset_mean = np.array(data_stats[0],dtype=np.float32)
        self.dataset_std = np.array(data_stats[1],dtype=np.float32)

    def get_all_mris_in_dataset(self):
        mri_folders = glob.glob(f"{self.data_dir}**/{self.mri_prefix}*/",recursive=True)
        mri_folders = self.remove_incomplete_mris(mri_folders)
        scan_dic = {fp.split("/")[-2]:fp for fp in mri_folders}
        print(f"Found {len(mri_folders)} MRIs")
        return list(scan_dic.keys()),scan_dic

    def remove_incomplete_mris(self,mri_folders):
        #if there are any you want to ignore just add them to this list
        removed_mris=[]
        return [fp for fp in mri_folders if fp.split("/")[-2] not in removed_mris]

    def swap_labels(self,label_data):
        uniques = np.unique(label_data)
        for u in uniques:
            if u not in [0, 1, 2, 4]:
                raise RuntimeError('unexpected label')

        new_label_data = np.zeros_like(label_data,dtype=np.int16)
        new_label_data[label_data == 4] = LABEL_MAP[4]
        new_label_data[label_data == 2] = LABEL_MAP[2]
        new_label_data[label_data == 1] = LABEL_MAP[1]
        return new_label_data

    def get_standardized_image(self, scan_full_path):
        image_data = nifti_io.read_in_patient_sample(scan_full_path,self.modality_extensions)
        crop_idxs = determine_brain_crop(image_data)
        cropped_data = image_data[crop_idxs]
        if(self.label_extension):
            label_data = nifti_io.read_in_labels(scan_full_path,self.label_extension)
            cropped_labels = label_data[crop_idxs]
            standardized_labels=self.swap_labels(cropped_labels)
        else:
            standardized_labels=None


        normalized_data = normalize_img(cropped_data)
        standardized_data = standardize_img(normalized_data,self.dataset_mean,self.dataset_std)

        return standardized_data,standardized_labels,crop_idxs

    #returns the mean and standard deviation of all MRIs
    def compute_dataset_stats(self):
        print("Computing dataset mean and SD")
        img_means=[]
        img_deviations=[]
        for mri_id in self.all_ids:
            mri_path=self.id_to_fp[mri_id]
            img = nifti_io.read_in_patient_sample(mri_path,self.modality_extensions)
            lab = nifti_io.read_in_labels(mri_path,self.label_extension)
            if(len(img.shape)>3):
                healthy_tissue_mask = np.logical_and(img[:,:,:,0]>0.001,lab==0)
                img=img[healthy_tissue_mask]
                img = normalize_img(img,is_flat=True)
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


    def process_next_sample(self,mri_id):
        image_data,label_data,crop_idxs = self.get_standardized_image(self.id_to_fp[mri_id])
        nx_graph,node_feats,region_img = img2graph(image_data,label_data,self.num_nodes,self.boxiness_coef,self.num_neighbors)
        #save in correct folder
        save_path = f"{self.output_dir}{os.sep}{mri_id}"
        if not os.path.exists(save_path):
            print("making dir",save_path)
            os.makedirs(save_path)
        graph_io.save_networkx_graph(nx_graph, f"{save_path}{os.sep}{mri_id}_nxgraph.json")
        nifti_io.save_as_nifti(image_data,f"{save_path}{os.sep}{mri_id}_input.nii.gz")
        nifti_io.save_as_nifti(label_data,f"{save_path}{os.sep}{mri_id}_label.nii.gz")
        nifti_io.save_as_nifti(region_img,f"{save_path}{os.sep}{mri_id}_supervoxels.nii.gz")
        np.save(f"{save_path}{os.sep}{mri_id}_crop.npy",crop_idxs)
        return mri_id

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(self.process_next_sample,mri_id) for mri_id in self.all_ids]
            print("Set up Threads, starting execution")
            for future in concurrent.futures.as_completed(futures):
                try:
                    mri_id = future.result()
                except Exception as exc:
                    print(f"Thread generated exception {exc}")
                else:
                    print("Finished "+ mri_id)


if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default=None, help='path to the directory where data is stored',type=str)
    parser.add_argument('-n', '--num_nodes', default=15000, help='How many supervoxels to segment brain into',type=int)
    parser.add_argument('-k', '--num_neighbors', default=10, help='How many neighbors each node has in the adjacency matrix',type=int)
    parser.add_argument('-b', '--boxiness', default=0.5, help='How square (regular) the supervoxels should be (recommended range=[0.1,1.0])',type=float)
    parser.add_argument('-o', '--output_dir', default=None,help='Directory to save graphs to',type=str)
    parser.add_argument('-m','--modality_extensions', nargs="+", default=["_flair.nii.gz","_t1.nii.gz","_t1ce.nii.gz","_t2.nii.gz"],help="The file extensions for each desired modality. Accepts a variable amount of modalities. Ensure consistent order.")
    parser.add_argument('-l', '--label_extension', default=None, help='What the label extension is. At evaluation ignore')
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.')

    args = parser.parse_args()
    gen = DataPreprocessor(args)
    gen.run()