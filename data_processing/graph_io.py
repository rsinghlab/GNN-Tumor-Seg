import numpy as np
import os
import networkx as nx
import pickle
import json
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm



#assign each voxel the label of its supervoxel
def project_nodes_to_img(svs,node_labels):
    #the background is -1 in the sv partitioning, so set this to be healthy
    node_labels = np.append(node_labels,0)
    return node_labels[svs]


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
    #print("Saved ",fp)

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
