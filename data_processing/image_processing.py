import numpy as np
from scipy import ndimage

def determine_tumor_crop(preds):
    mask = preds!=0
    #make it a bit bigger
    mask = ndimage.binary_dilation(mask)
    ix = np.ix_(mask.any(axis=(1,2)),mask.any(axis=(0,2)),mask.any(axis=(0,1)))
    return ix

#uncrops image back to full size (fills space around with healthy preds)
def uncrop_to_brats_size(mri_id,dataset,voxel_preds):
    brain_crop = dataset.brain_crops[mri_id]
    full_img_shape = dataset.img_shapes[mri_id]
    full_img = np.zeros(full_img_shape)
    full_img[brain_crop]=voxel_preds
    return full_img


#Creates closest crop possible by discarding all planes that are entirely black
#Note returns boolean array, does NOT actually perform crop
#this is so you can also crop the labels
def determine_brain_crop(multi_modal_data):
    if(len(multi_modal_data.shape)==4):
        max_intensity_vals = np.amax(multi_modal_data,axis=3)
    elif(len(multi_modal_data.shape)==3):
        max_intensity_vals = multi_modal_data
    else:
        raise Exception(f"Expected input shape of either nxmxr or nxmxrxC. Instead got {multi_modal_data.shape}")
    mask = max_intensity_vals>0.01
    ix = np.ix_(mask.any(axis=(1,2)),mask.any(axis=(0,2)),mask.any(axis=(0,1)))

    return ix


#put each modality in approx 0,1 range. not sure what the brats units are but they dont seem to have a cap
def normalize_img(img_array,is_flat=False):
    if(is_flat):
        maxes = np.quantile(img_array,0.995,axis=0).astype(np.float32)
    else:
        maxes = np.quantile(img_array,0.995,axis=(0,1,2)).astype(np.float32)
    #print("Max value for each modality", maxes)
    return img_array/maxes

def standardize_img(img_array,mean,std):
    centered = img_array-mean
    standardized = centered/std
    return standardized