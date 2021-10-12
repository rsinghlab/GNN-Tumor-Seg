import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from data_processing.nifti_io import read_nifti
from data_processing.image_processing import normalize_img



#to plot the supervoxels use this cmap so it displays in a readable manner
def define_cluster_cmap(supervoxel_partitioning):
    num_segments = len(np.unique(supervoxel_partitioning))-1
    lut = np.random.rand(num_segments, 3)
    lut=np.insert(lut,0,(0,0,0),axis=0)
    cmap = LinearSegmentedColormap.from_list('cluster_map', lut, N=num_segments)
    return cmap

#set continuous labels to true if you use the remapping 0,1,2,3 rather than BraTS 0,1,2,4
def define_label_cmap(continuous_labels=False):
    if continuous_labels:
        num_segments = 4
        lut = np.array([
            [50, 168, 82], #green
            [219, 13, 41], #red
            [13, 51, 219], #blue
            [219, 185, 13] #yellow
            ])/255
    else:
        num_segments = 5
        lut = np.array([
            [50, 168, 82],
            [13, 51, 219],
            [219, 13, 41],
            [0, 0, 0],
            [219, 185, 13]])/255
    print("Healthy: Green (or transparent), Edema:Red, NET:Blue, ET: Yellow")
    cmap = LinearSegmentedColormap.from_list('label_map', lut, N=num_segments)
    return cmap, lut


#should modify in place
def plot_three_on_axis(ax,i1,i2,i3):
    ax[0].imshow(i1,cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(i2,cmap='gray')
    ax[1].axis('off')
    ax[2].imshow(i3,cmap='gray')
    ax[2].axis('off')

def plot_four_on_axis(ax,i1,i2,i3,i4):
    ax[0].imshow(i1,cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(i2,cmap='gray')
    ax[1].axis('off')
    ax[2].imshow(i3,cmap='gray')
    ax[2].axis('off')
    ax[3].imshow(i4,cmap='gray')
    ax[3].axis('off')


#move to data processing
def get_data_for_plotting(data_folder,seg_folder,mri_id,mod1_ext,mod2_ext,read_labels=True):
    cmap,lut = define_label_cmap(True)
    mod1 = read_nifti(f"{data_folder}/{mri_id}/{mri_id}{mod1_ext}",np.float32)
    mod1 = mod1/np.max(mod1)
    mod2 = read_nifti(f"{data_folder}/{mri_id}/{mri_id}{mod2_ext}",np.float32)
    mod2 = mod2/np.max(mod2)
    modality_to_overlay = np.copy(mod1)
    predictions = read_nifti(f"{seg_folder}/{mri_id}.nii.gz",np.int16)

    #make rgb (three channel)
    overlaid_preds = np.stack([modality_to_overlay for i in range(3)],-1)
    #just have a dummy array of correct size
    overlaid_gt = np.zeros_like(overlaid_preds)

    if(read_labels):
        labels = read_nifti(f"{data_folder}/{mri_id}/{mri_id}_seg.nii.gz",np.int16)
        overlaid_gt = np.stack([modality_to_overlay for i in range(3)],-1)

    #overlay values
    for i in range(1,len(lut)):
        overlaid_preds[predictions==i]=lut[i]
        if(read_labels):
            overlaid_gt[labels==i]=lut[i]

    #zoom in a bit, the vast majority of brains are just background at these coords
    overlaid_preds=overlaid_preds[30:220,30:220]
    mod1=mod1[30:220,30:220]
    mod2=mod2[30:220,30:220]
    overlaid_gt=overlaid_gt[30:220,30:220]

    return mod1,mod2,overlaid_preds,overlaid_gt
