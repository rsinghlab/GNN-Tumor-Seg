import numpy as np
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from scipy import ndimage
import networkx as nx
from collections import defaultdict
from time import time


'''
Contains the code for converting an image to a graph. The general idea is to use SLIC to partition an image into a set of supervoxels
which are treated as nodes and then construct edges between adjacent supervoxels.
The img2graph function does this for a given numpy array by invoking the remaining helper functions in this file.
'''


def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    m = counts.argmax()
    return values[m]


def get_quantiles(x):
    quants = np.quantile(x, [0.1,0.25,0.5,0.75,0.9])
    return quants
    

#Combines the functionality of determining labels and features for each supervoxel with discarding supervoxels that lie outside of the brain.
def determine_nodes_and_features(supervoxel_partitioning,voxel_intensities,voxel_labels,num_supervoxels):
    sv_feats,sv_centroids,sv_labels = extract_supervoxel_statistics(supervoxel_partitioning,voxel_intensities,voxel_labels,num_supervoxels)
    new_region_img,node_feats,node_centroids,node_labels = discard_empty_svs(supervoxel_partitioning,sv_feats,sv_centroids,sv_labels,num_supervoxels)
    return new_region_img,node_feats,node_centroids,node_labels

'''
For each supervoxel determine its centroid, most common label in the label image, and extract summary statistics from the image.
The summary statistics are then assigned as features to each supervoxels corresponding node.

sv_partitioning: The output of SLIC

voxel_intensities: The (multichannel) intensities of each voxel as a numpy array, with the channel information being in the last dimension.

voxel_labels: Numpy array of integers of the same spatial shape as voxel_intensities and sv_partitioning.

'''
def extract_supervoxel_statistics(sv_partitioning,voxel_intensities,voxel_labels,num_supervoxels):
    #first check how many incoming channels there are
    if(len(voxel_intensities.shape)>3):
        num_modalities = voxel_intensities.shape[-1]
        sv_feats = []
        for i in range(num_modalities):
            sv_feats.append(get_sv_summary_for_modality(voxel_intensities[:,:,:,i],sv_partitioning,num_supervoxels))
        sv_feats=np.concatenate(sv_feats,axis=-1)
    else:
        #only 1 channel
        sv_feats = get_sv_summary_for_modality(voxel_intensities,sv_partitioning,num_supervoxels)
    
    sv_labels = ndimage.labeled_comprehension(voxel_labels,labels=sv_partitioning,func=mode,index=range(0,num_supervoxels),out_dtype='int32',default=-1.0)

    #centroid=center of mass where the mass is equally distributed *taps forehead*
    sv_centroids = np.array(ndimage.center_of_mass(np.ones(sv_partitioning.shape),sv_partitioning,range(0,num_supervoxels)))
    return sv_feats,sv_centroids,sv_labels


'''
There is a considerable amount of "dead" space in an MRI image, since the the image is a rectangular prism but the brain is an ellipsoid.
SLIC thus produces many supervoxels that are composed entirely of background, black voxels. Since these aren't part of the brain,
we dont want to include them in our graphical model thereof and we instead discard them.
This requires us to renumber the supervoxels to ensure that the numbering remains continuous and matches the extracted
features, labels, and centroids.
'''
def discard_empty_svs(sv_partitioning,sv_features,sv_centroids,sv_labels,n_svs):
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
    new_region_img = new_regions[sv_partitioning]
    node_feats = sv_features[np.invert(empty_mask)]
    node_centroids = sv_centroids[np.invert(empty_mask)]
    node_labels = sv_labels[np.invert(empty_mask)]

    return new_region_img,node_feats,node_centroids,node_labels

'''
Given an array of intensities and a supervoxel partitioning of the same shape, determines summary statistics of the intensities
grouped by the supervoxels. In this case returns quintiles for each supervoxel but could just as easily be the mean or a histogram, etc.
If the image is partitioned into 500 supervoxels, will return a vector of 500 quintiles as a 2D numpy array.

Must be called seperately for each modality since the labeled_comprehension function does not, afaik, support multichannel computation.
'''
def get_sv_summary_for_modality(modality_intensities,sv_partitioning,n_svs):
    sv_feats = ndimage.labeled_comprehension(modality_intensities,labels=sv_partitioning,func=get_quantiles,index=range(0,n_svs),out_dtype='object',default=-1.0)
    sv_feats = np.stack(sv_feats,axis=0)
    return sv_feats


'''
Build an adjacency matrix such that each supervoxel has approximately k adjacent supervoxels.

positions: The spatial positions of the supervoxels (typically centroids)

intensities: The features of each supervoxel. This is only used to compute edge weights.

k: How many approximate neighbors each node should have.

weighted: Whether to return a boolean or weighted adjacency matrix. Some graph learning algorithms can handle edge weights.

enforce_regularity: If True, all but the last handful of nodes will have exactly k edges. If False, each node will have at least k edges.
That is, if False, each supervoxel will be assigned the k supervoxels closest to it as neighbors. However, another supervoxel that is not one of those k
may assign *it* as a neighbor if *it* is one the of k closest to that other supervoxel. Since edges are undirected, the first node will now have k+1 neighbors.
'''
def build_adjacency_matrix(positions,intensities,k,weighted=True,enforce_regularity=True):
    euc_distances = cdist(positions,positions,metric='euclidean')
    adjacency_matrix = np.zeros(euc_distances.shape)

    #slower but ensures that each node has (almost) exactly k neighbors, not just outgoing edges
    #some nodes will end up with a different amount of edges but nearly all will have k
    if(enforce_regularity):
        top_k_arr = np.argsort(euc_distances,axis=1)
        for i in range(len(top_k_arr)):
            possible_neighbors = top_k_arr[i][top_k_arr[i]>i]
            num_neighbors_needed = int(k - np.sum(adjacency_matrix[i]))
            if(num_neighbors_needed>0):
                neighbors_to_add = possible_neighbors[:num_neighbors_needed]
                adjacency_matrix[i][neighbors_to_add]=1
                #also add the transpose
                adjacency_matrix[neighbors_to_add,i]=1

    else:
        top_k_arr = np.argsort(euc_distances,axis=1)[:,:k]
        for i in range(len(top_k_arr)):
            adjacency_matrix[i][top_k_arr[i]]=1

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

'''
Determines which supervoxels are adjacent to which other supervoxels.
regionImg is a supervoxel partitioning (either 2D or 3D numpy array of integers)
n_nodes is the number of supervoxels, equivalent to len(np.unique(regionImg))
Can return results either as an adjacency matrix or as an edge set.
'''
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



'''
Given an image (numpy array) returns a networkx graph with node features.

#Input#
voxel_intensities: a 3D or 4D numpy array (typically a MRI image). Assumed to have at least three spatial dimensions. If fourth dimension is present,
the last one is assumed to be channel information.

voxel_labels: 3D numpy array of non-negative integers. Should be same shape as voxel_intensities in the spatial dimensions. Can be None. 
If present assumed to be the label for each voxel.

approx_num_nodes: Number of supervoxels SLIC should partition brain into. This is only approximate. Note that the final number of nodes
in the graph will be much smaller (less than half) since supervoxels created in the empty space around the brain are later filtered out.

boxiness: a parameter for the SLIC algorithm (compactness coefficient). Higher values will lead to cubic supervoxels, smaller values will
lead to more irregular supervoxels (see SLIC documentation)

k: The number of neighbors (undirected edges) each node in the graph should have. If None, each supervoxel will have an edge with every
adjacent (touching) supervoxel. This creates an irregular graph which some graph learning algorithms cannot operate on.


#Output#
nx_graph: A networkx graph where each node has a feature vector and a label.

sv_feats: the features for each node as a numpy array

updated_partitioning: A numpy array of integers with the same spatial shape as the input image. Each voxel is assigned the integer of
the supervoxel it belongs to e.g.
-1-1-1-1
 0 0 1 -1
 0 1 1 -1
 2 2 1 -1
This simplified 2D representation has 3 supervoxels (0,1,2). -1 is reserved for empty(black) voxels outside of the brain. 
The original image can be reconstructed by assigning the features of each node to the corresponding voxels in the partitioning.
Similarly, one the nodes have been classified, the predictions can be reverted to an image by assigning each node's label
to its constituent voxels. E.g. if supervoxel 0 is predicted to be active tumor (tumor label 3) 
then label 3 will be projected onto all the voxels which are assigned the number 0.



'''
def img2graph(voxel_intensities,voxel_labels,approx_num_nodes=5000,boxiness=0.5,k=10):
    labels_provided = True if voxel_labels is not None else False
    multi_channel = True if len(voxel_intensities.shape)==4 else False
    slic_partitioning = slic(voxel_intensities.astype(np.float64),n_segments=approx_num_nodes,sigma=1,compactness=boxiness,multichannel=multi_channel,convert2lab=False).astype(np.int16)
    num_supervoxels = np.amax(slic_partitioning)+1
    #print("Number of supervoxels generated by SLIC: ", num_supervoxels)

    if(not labels_provided):
        #create pseudo-labels
        voxel_labels=np.zeros(voxel_intensities.shape[:3],dtype=np.int16)

    updated_partitioning,sv_feats,sv_centroids, sv_labels = determine_nodes_and_features(np.copy(slic_partitioning),voxel_intensities,voxel_labels,num_supervoxels)

    
    if(k):
        graph_adjacency = build_adjacency_matrix(sv_centroids,sv_feats,k,weighted=False,enforce_regularity=True)
    else:
        graph_adjacency = find_adjacent_nodes(updated_partitioning,len(sv_labels),as_mat=True)

    nx_graph = nx.from_numpy_matrix(graph_adjacency)
    for n in nx_graph.nodes:
        if(labels_provided):
            label = int(sv_labels[n])
            nx_graph.nodes[n]["label"]=label
        features = list(sv_feats[n])
        nx_graph.nodes[n]["features"]=features

    return nx_graph,sv_feats,updated_partitioning

