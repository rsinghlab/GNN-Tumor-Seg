import nibabel as nib
import numpy as np
import os

'''
A collection of helper functions to read and write niftii files
'''

#Given a directory and a list of modality extensions, scans that directory for any files ending in those extensions
#and returns them as a single stacked numpy float array.
#Assumes the modalities all belong to the same patient, have already been co-registered, and have the same resolution.
def read_in_patient_sample(scan_dir,modality_exts):
    num_modalities=len(modality_exts)
    modality_imgs = []
    for root, _, files in os.walk(scan_dir):
        for ext in modality_exts:
            for filename in files:
                if filename.endswith(ext):
                    filepath = os.path.join(root, filename)
                    mod_img = nib.load(filepath)
                    #data is actually stored as int16
                    img_data = np.array(mod_img.dataobj,dtype=np.float32)
                    modality_imgs.append(img_data)
    #check that all the modalities were present in the folder
    assert(len(modality_imgs)==num_modalities)

    patient_sample = np.stack(modality_imgs,3) if num_modalities>1 else modality_imgs[0]
    return patient_sample


def read_in_labels(scan_dir,label_ext):
    for filename in os.listdir(scan_dir):
        if filename.endswith(label_ext):
            label_nib = nib.load(scan_dir+os.sep+filename)
            #potentially also return affine if they are different between images (which they are not for brats)
            return np.array(label_nib.dataobj,dtype=np.int16)
    raise FileNotFoundError(f"Label image not found in folder: {scan_dir}")


#Uses the BraTS standard affine matrix to save a numpy array into a nifti file.
#This will likely break any data that is not curated by BraTS or TCIA
def save_as_nifti(img,fp):
    affine_mat = np.array([
        [ -1.0,  -0.0,  -0.0,  -0.0],
        [ -0.0,  -1.0,  -0.0, 239.0],
        [  0.0,   0.0,   1.0,   0.0],
        [  0.0,   0.0,   0.0,   1.0],
        ])
    img = nib.nifti1.Nifti1Image(img, affine_mat)
    nib.save(img, fp)

def read_nifti(fp,data_type):
    nib_obj = nib.load(fp)
    return np.array(nib_obj.dataobj,dtype=data_type)



