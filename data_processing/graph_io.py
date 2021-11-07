import numpy as np
import os
import networkx as nx
import pickle
import json

'''
A collection of helper functions for graph processing
'''


'''
Input:
A supervoxel partitioning(2D or 3D array of integers)
A vector of labels for each node of the same length as the number of unique integers in the supervoxel partitioning (excl -1)
Function:
Assigns each voxel the label of the supervoxel it belongs to. E.g. if a voxel has the value 100, it will be assigned the label
at index 100 in the vector of node_labels.
Voxels labelled with -1 are assigned a label of 0.
'''
def project_nodes_to_img(svs,node_labels):
    #the background is -1 in the sv partitioning, so set this to be healthy
    node_labels = np.append(node_labels,0)
    return node_labels[svs]


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


#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

# def draw_graph(G):
#     node_colors = '#1f78b4'
#     if 'label' in G.nodes[0]:
#         node_colors = []
#         for node in G.nodes:
#             if G.nodes[node]['label']==0:
#                 node_colors.append(0.3)
#             elif G.nodes[node]['label']==1:
#                 node_colors.append(0.7)
#             elif G.nodes[node]['label']==2:
#                 node_colors.append(0)
#             elif G.nodes[node]['label']==4:
#                 node_colors.append(1)
#             else:
#                 raise Exception("not a possible label")

#     fig, ax = plt.subplots(figsize=(20,20))
#     nx.draw(G,ax=ax,node_color=node_colors,alpha=0.6)
#     plt.show()
