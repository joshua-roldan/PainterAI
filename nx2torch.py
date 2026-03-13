import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import copy
import networkx as nx
import torch
from operator import itemgetter


""" ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
             Graph Conversion Functions
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! """

# These functions are called primarily to convert between networkx functionalities 
# to Pytorch. It is organized such that all information is stored in the networkx graph,
# avoiding any call to the "game state" itself.
# The locations L for where we need to convert are:
#  L.1. "Agent.py / get_action(state, game) / step 2.B. (Exploitation) " 
#      so the GNN outputs a prediction.
#  L.2. "Model.py / train_step(state, next_state, ...) / step 1 "
#      so the GNN learns the colored graphs for both "state" and "next_state".

" -------- Convert from NetworkX to PyTorch -------- "

def convert_nodes_from_networkx_to_pytorch(networkx_graph):
    # Convert an integer representative of a node color into 
    # a 1-hot vector encoding.
    node_colors        = []
    ncolors            = networkx_graph.graph['num_of_colors']
    color_sku_template = [0] * ncolors
    for n in networkx_graph.nodes:
        color_sku = copy.deepcopy(color_sku_template)
        color_int = networkx_graph.nodes[n]['color']
        color_sku[color_int-1] = 1
        node_colors.append(color_sku)
    node_colors  = torch.tensor(np.asarray(node_colors), dtype=torch.float)
    return node_colors

def convert_edges_from_networkx_to_pytorch(networkx_graph):
    node_dict = networkx_graph.graph['node_dict']
    edge_list = [ [node_dict[e[0]], node_dict[e[1]]] for e in networkx_graph.edges ]
    #print('edge_list: ', edge_list)
    edge_list   = torch.tensor(np.asarray(edge_list).T, dtype=torch.long) 
    if edge_list.nelement() == 0 :
        edge_list = torch.tensor(np.array([[0], [0]]))
    return edge_list

    
""" ------- Convert from PyTorch to NetworkX  -------- """

def convert_node_idx_to_label(idxN, networkx_graph):
    #print('node_idx: ', idxN)
    #print('networkx_graph.nodes:', networkx_graph.nodes)
    #print('node dict: ', nx_graph_old.graph['node_dict'])
    node_label = list(networkx_graph.nodes)[idxN]
    return node_label

def convert_node_sku_to_label(node_sku):
    # Converts the GNN outputted node sku into its corresponding node label in the networkx graph.
    # The input 'node_sku' is formatted as a vector. 
    # In the 'nx.grid" graph, the nodes are named (i,j) 
    # where i is the row and j is the columnn. 
    # To faciliate conversion between vector to (i,j) we use np.reshape().
    dim1, dim2 = 10, 10
    # dim1, dim2 = grid_dimensions[0], grid_dimensions[1]
    node_sku = np.reshape(node_sku, (dim1, dim2))
    print('node_sku: ', node_sku)
    idxs = (np.argwhere(node_sku==1))[0]
    rowidx, colidx = idxs[0], idxs[1]
    print('rowidx: ', rowidx)
    print('colidx: ', colidx)
    node_label = (rowidx, colidx)
    return node_label

def convert_color_sku_to_int(color_sku):
    # The GNN takes and sends vector inputs and outputs, respectively,
    # but the networkx graph nodes color attribute is an integer.
    color_int = 1+ ((np.argwhere(np.asarray(color_sku)==1))[0][0])
    return color_int

def convert_sku_to_int(sku):
    # The GNN takes and sends vector inputs and outputs, respectively,
    # but the networkx graph nodes attribute are integers (or pair of integers).
    int = ((np.argwhere(np.asarray(sku)==1))[0][0])
    return int

def plus_two(x):
    x += 2
    return x