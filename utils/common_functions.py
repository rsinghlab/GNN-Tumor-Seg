import numpy as np
from scipy import ndimage




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
def normalize_img(img_array):
    maxes = np.quantile(img_array,0.995,axis=(0,1,2))
    #print("Max value for each modality", maxes)
    return img_array/maxes

def standardize_img(img_array,mean,std):
    centered = img_array-mean
    standardized = centered/std
    return standardized


#Creates the adjacency matrix based on actual adjacency
def find_adjacent_nodes(regionImg,n_nodes,as_mat=False):

    #first replace -1 with the next largest region number (i.e. the current number of nodes because 0 indexed) to ease computation
    regionImg[regionImg==-1]=n_nodes
    tmp = np.zeros((n_nodes+1, n_nodes+1), bool)

    if(len(regionImg.shape)==3):
        # check the vertical adjacency
        a, b = regionImg[:-1, :, :], regionImg[1:, :, :]
        tmp[a[a!=b], b[a!=b]] = True
        # check the horizontal adjacency
        a, b = regionImg[:, :-1, :], regionImg[:, 1:, :]
        tmp[a[a!=b], b[a!=b]] = True
        # check the depth adjacency
        a, b = regionImg[:, :, :-1], regionImg[:, :, 1:]
        tmp[a[a!=b], b[a!=b]] = True
    #2D case
    else:
        a, b = regionImg[:-1, :], regionImg[1:, :]
        tmp[a[a!=b], b[a!=b]] = True
        a, b = regionImg[:, :-1], regionImg[:, 1:]
        tmp[a[a!=b], b[a!=b]] = True

    # register adjacency in both directions (up, down) and (left,right)
    adj_mat = (tmp | tmp.T)

    #exclude last row and column (-1 areas)
    adj_mat = adj_mat[:-1,:-1]
    np.fill_diagonal(adj_mat,True)
    #revert changes to regionImg
    regionImg[regionImg==n_nodes]=-1

    #return results as adjacency matrix
    if(as_mat):
        return adj_mat
    return np.where(adj_mat)

