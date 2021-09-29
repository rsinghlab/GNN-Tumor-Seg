import numpy as np
import os
import networkx as nx
import pickle
import json
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm



#Method to read in clustering and assign each voxel the label of its supervoxel
def convert_supervoxel_labels_to_voxel(path_to_mri,sv_predictions=None,classify_background_as_healthy=True):
    voxel_sv_assignments = load_segmentation_array(path_to_mri)
    if sv_predictions is None:
        #array of labels (presumably in order of sv number)
        sv_predictions = read_supervoxel_predictions(path_to_mri)
    #cluster_to_pred_map = {i:sv_predictions[i] for i in range(len(sv_predictions))}
    cluster_to_pred_mapper = lambda cl: sv_predictions[cl] if cl!=-1 else -1
    voxel_predictions = np.vectorize(cluster_to_pred_mapper,otypes=[np.float64])(voxel_sv_assignments)
    if(classify_background_as_healthy):
        voxel_predictions=np.where(voxel_predictions==-1,0,voxel_predictions)
    return voxel_predictions




#after clustering into supervoxels, each one is assigned a label, this will return the labelling the brain is actually trained on
#I will parse this out from the graph to avoid having to save another file with this info
#Also prints the number of supervoxels with x label
#I padd the slice clustering to ease computation to just the slice of interest 
def get_supervoxel_labelling(path_to_mri,slice_clustering):
    G = load_networkx_graph(path_to_mri)
    #finding total labelled supervoxels (not related to functionality)
    #TODO: Make sure this is in the correct order
    sv_label_dict = {n:G.nodes[n]["label"] for n in G.nodes}
    sv_label_count_tuples = zip(*np.unique(list(sv_label_dict.values()),return_counts=True))
    print("Expected Supervoxel Labels",{l:c for l,c in sv_label_count_tuples})

    sv_label_dict[-1]=0
    slice_labelling = np.copy(slice_clustering)
    for k, v in sv_label_dict.items(): slice_labelling[slice_clustering==k] = v
    return slice_labelling




def save_networkx_graph(G,fp):
    graph_as_json = nx.readwrite.json_graph.node_link_data(G)
    str_dump = json.dumps(graph_as_json)
    with open(fp,'w') as f:
        f.write(str_dump)
    print("Saved ",fp)

def load_networkx_graph(fp):
    with open(fp,'r') as f:
        json_graph = json.loads(f.read())
        return nx.readwrite.json_graph.node_link_graph(json_graph)



def draw_graph(G):
    node_colors = '#1f78b4'
    if 'label' in G.nodes[0]:
        node_colors = []
        for node in G.nodes:
            if G.nodes[node]['label']==0:
                node_colors.append(0.3)
            elif G.nodes[node]['label']==1:
                node_colors.append(0.7)
            elif G.nodes[node]['label']==2:
                node_colors.append(0)
            elif G.nodes[node]['label']==4:
                node_colors.append(1)
            else:
                raise Exception("not a possible label")

    fig, ax = plt.subplots(figsize=(20,20))
    nx.draw(G,ax=ax,node_color=node_colors,alpha=0.6)
    plt.show()
