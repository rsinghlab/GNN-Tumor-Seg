import numpy as np
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from scipy import ndimage
import networkx as nx
from collections import defaultdict
from time import time





def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    m = counts.argmax()
    return values[m]


def get_quantiles(x):
    quants = np.quantile(x, [0.1,0.25,0.5,0.75,0.9])
    return quants
    

########PATH 1#########


def determine_node_feats_fast(voxel_clustering,voxel_intensities,voxel_labels,num_supervoxels):
    sv_feats,sv_centroids,sv_labels = find_supervoxel_feats(voxel_clustering,voxel_intensities,voxel_labels,num_supervoxels)
    new_region_img,node_feats,node_centroids,node_labels = discard_empty_svs(voxel_clustering,sv_feats,sv_centroids,sv_labels,num_supervoxels)
    return new_region_img,node_feats,node_centroids,node_labels

def get_sv_summary_for_modality(modality_intensities,region_img,n_svs):
    sv_feats = ndimage.labeled_comprehension(modality_intensities,labels=region_img,func=get_quantiles,index=range(0,n_svs),out_dtype='object',default=-1.0)
    sv_feats = np.stack(sv_feats,axis=0)
    return sv_feats


def find_supervoxel_feats(region_img,voxel_intensities,voxel_labels,num_supervoxels):
    #first check how many incoming channels there are
    if(len(voxel_intensities.shape)>3):
        num_modalities = voxel_intensities.shape[-1]
        sv_feats = []
        for i in range(num_modalities):
            sv_feats.append(get_sv_summary_for_modality(voxel_intensities[:,:,:,i],region_img,num_supervoxels))
        sv_feats=np.concatenate(sv_feats,axis=-1)
    else:
        #only 1 channel
        sv_feats = get_sv_summary_for_modality(voxel_intensities,region_img,num_supervoxels)
    
    sv_labels = ndimage.labeled_comprehension(voxel_labels,labels=region_img,func=mode,index=range(0,num_supervoxels),out_dtype='int32',default=-1.0)

    #centroid=center of mass where the mass is equally distributed *taps forehead*
    sv_centroids = np.array(ndimage.center_of_mass(np.ones(region_img.shape),region_img,range(0,num_supervoxels)))
    return sv_feats,sv_centroids,sv_labels


#if a SV is composed entirely out of dark voxels, this implies that it is empty, and we exclude it
def discard_empty_svs(region_img,sv_features,sv_centroids,sv_labels,n_svs):
    #empty supervoxels are black, which means their max intensity in one modality should be similar to global min in that modality
    #index = 4 because this is the highest quintile of the first modality
    global_min = np.min(sv_features[:,4])
    empty_mask = sv_features[:,4]<global_min+0.01
    #renumber regions
    new_regions = np.zeros(n_svs,dtype=np.int16)
    n_nodes=0
    for i in range(n_svs):
        if(empty_mask[i]):
            new_regions[i]=-1
        else:
            new_regions[i]=n_nodes
            n_nodes+=1
    new_region_img = new_regions[region_img]
    node_feats = sv_features[np.invert(empty_mask)]
    node_centroids = sv_centroids[np.invert(empty_mask)]
    node_labels = sv_labels[np.invert(empty_mask)]

    return new_region_img,node_feats,node_centroids,node_labels


#######ADJACENCY STUFF #######

#can return a boolean adjacency matrix or weighted
def build_adjacency_matrix(positions,intensities,k,weighted=True,enforce_regularity=False):
    euc_distances = cdist(positions,positions,metric='euclidean')
    adjacency_matrix = np.zeros(euc_distances.shape)

    #slower but ensures that each node has exactly k neighbors, not just outgoing edges
    if(enforce_regularity):
        #this one is almost correct
        #sometimes runs out of possible edges to add?
        top_k_arr = np.argsort(euc_distances,axis=1)
        for i in range(len(top_k_arr)):
            possible_neighbors = top_k_arr[i][top_k_arr[i]>i]
            num_neighbors_needed = int(k - np.sum(adjacency_matrix[i]))
            if(num_neighbors_needed>0):
                neighbors_to_add = possible_neighbors[:num_neighbors_needed]
                adjacency_matrix[i][neighbors_to_add]=1
                #also add the transpose
                adjacency_matrix[neighbors_to_add,i]=1
        #figure out some way to reroute edges from nodes with too many neighbors to those with too little.


    else:
        top_k_arr = np.argsort(euc_distances,axis=1)[:,:k]
        for i in range(len(top_k_arr)):
            adjacency_matrix[i][top_k_arr[i]]=1
    #anotehr way of doing this is to say you can only add a node as a neighbor if it comes afer you, or you already share an edge
    if weighted:
        intensity_dists = cdist(intensities,intensities,metric='euclidean')
        #normalize dists to be between 0 and 1
        intensity_dists = intensity_dists/np.amax(intensity_dists)
        #Apply formula
        sigma = 0.1
        #sigma = np.median(intensity_dists,axis=None)
        edge_weights = np.exp(-(intensity_dists**2)/(2*sigma**2))
        adjacency_with_weights = edge_weights*adjacency_matrix
        return adjacency_with_weights
    else:
        return adjacency_matrix


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




#Where the images are located (expected to be stored as .nii.gz)
# the approximate number of supervoxels the SLIC algorithm should generate
# whether to return the result as a networkx graph (the default is features,labels,adjacency set)
#if k = None and networkxz is True, then adjacent supervoxels will be considered neighbors
#if cropped_labels is None, then don't save label information
def img2graph(voxel_intensities,voxel_labels,approx_num_nodes=5000,boxiness=0.5,k=10):
    labels_provided = True if voxel_labels is not None else False
    multi_channel = True if len(voxel_intensities.shape)==4 else False
    slic_clustering = slic(voxel_intensities.astype(np.float64),n_segments=approx_num_nodes,sigma=1,compactness=boxiness,multichannel=multi_channel,convert2lab=False).astype(np.int16)
    num_supervoxels = np.amax(slic_clustering)+1
    #print("Number of supervoxels generated by SLIC: ", num_supervoxels)

    if(not labels_provided):
        #create pseudo-labels just so i dont have to change things further down the line
        voxel_labels=np.zeros(voxel_intensities.shape[:3],dtype=np.int16)

    updated_clustering,sv_feats,sv_centroids, sv_labels = determine_node_feats_fast(np.copy(slic_clustering),voxel_intensities,voxel_labels,num_supervoxels)

    
    if(k):
        graph_adjacency = build_adjacency_matrix(sv_centroids,sv_feats,k,weighted=False,enforce_regularity=True)
    else:
        graph_adjacency = find_adjacent_nodes(updated_clustering,len(sv_labels),as_mat=True)

    nx_graph = nx.from_numpy_matrix(graph_adjacency)
    for n in nx_graph.nodes:
        if(labels_provided):
            label = int(sv_labels[n])
            nx_graph.nodes[n]["label"]=label
        features = list(sv_feats[n])
        nx_graph.nodes[n]["features"]=features

    return nx_graph,sv_feats,updated_clustering

