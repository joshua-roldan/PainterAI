import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import copy
import networkx as nx
import torch
from operator import itemgetter
import random
import time


""" ------------- Indexer Object --------------- """

class Indexer():
    def __init__(self):
        self.current_index  = 0
        self.indexed_items = {}
    
    def increase_current_index(self):
        self.current_index += 1

    def index_new_item(self, item):
        self.indexed_items[item] = self.current_index
        self.increase_current_index()

    def check_index(self, key):
        keys = self.indexed_items.keys()
        if key not in keys:
            self.index_new_item(key)

    def sample_indexed_items(self, sample_keys):
        self.sample = {sk: self.indexed_items[item] for sk, item in enumerate(sample_keys)}
        return self.sample

    def reindex_items(self, indexed_items):
        sorted_index = sorted( indexed_items.items() , key=itemgetter(1) )
        new_index = {item: ii for (ii, (item, jj)) in enumerate( sorted_index )}
        return new_index
        
    def reset_index(self):
        self.current_index = 0
        self.index_dict    = {}


""" ------------ Item Counter ------------------ """

class ItemCounter():
    # Track a score or variable.
    def __init__(self):
        self.count = 0
        self.time_series = []      # Used to track an item during a given time window.
        self.long_time_series = [] # Used to track an item over many time windows.

    def add_items(self, number_of_items):
        self.count += number_of_items
        self.time_series.append(number_of_items)

    def average_count(self):
        mean = self.count/ len(self.time_series)
        return mean

    def reset_count(self):
        self.count = 0
        self.time_series = []



""" --------------- Glia Cell Helper ----------------"""

class GliaCell():
    # Probes the whole training process and stores information as to double check.
    def __init__(self):
        # A general notebook for probing things.
        self.notes = {}

    def approve_graph_construction(self, nx_graph, py_graph):
        assert nx_graph.number_of_nodes() == py_graph.x.shape[0], f"nx graph and py graph differ in number of nodes"
        print(f'  py graph'); 
        print(f'  nodes: {py_graph.x.shape}')
        print(f'  edges: {py_graph.edge_index.shape}')

    def approve_action(self, idxN, nx_graph, py_graph):
        assert 0    in list(range(nx_graph.number_of_nodes())), f" zero is not in the nx node indices"
        assert idxN in list(range(nx_graph.number_of_nodes())), f" idxN is not in the nx node indices"
        assert 0    in list(range(py_graph.x.shape[0]))       , f" zero is not in the py node indices"
        assert idxN in list(range(py_graph.x.shape[0]))       , f" idxN is not in the py node indices"

    def subgraph_into_py_graph(self, py_graph, boundary_index):
        boundary_nodes_idxs = torch.tensor([idx for (node, idx) in boundary_index])
        boundary_graph = torch.subgraph(boundary_nodes_idxs, py_graph.edge_index, py_graph.edge_attr)
        return boundary_graph

    def filter_out_graph_interior_from_prediction(self, np_prediction, boundary_index):
        boundary_nodes_idxs = [idxN for (idxBN, idxN) in boundary_index.items()]
        return np_prediction[boundary_nodes_idxs]


""" ---------------------------------- """
""" -------  Plotting  Functions ----- """
""" ---------------------------------- """

# check out: https://matplotlib.org/stable/gallery/animation/animate_decay.html#sphx-glr-gallery-animation-animate-decay-py

plt.ion() # interactive mode on

def visualize_colored_graph(fig_int, networkx_graph):

    plt.figure(fig_int, figsize=(5,8)); plt.clf()
    plt.title(f"Number of colors: {networkx_graph.graph['num_of_colors']}")
    
    # Larger graph
    plt.subplot(4,1,(1,2))
    plt.title("Underlying Graph")
    nx.draw_networkx(G=networkx_graph,
                    pos = {n: n for n in networkx_graph},
                    node_color= [networkx_graph.nodes[n]['color'] for n in networkx_graph.nodes],
                    with_labels=False,
                    vmin= 0,
                    vmax= networkx_graph.graph['num_of_colors'])
    
    # Subgraph
    plt.subplot(4,1,(3,4))
    plt.title("Painting")
    painting = networkx_graph.subgraph(networkx_graph.graph['painted_nodes']).copy()
    nx.draw_networkx(G=painting, 
                    pos = {n: n for n in painting},
                    node_color = [painting.nodes[n]['color'] for n in painting.nodes],
                    with_labels=False,
                    vmin=0,
                    vmax=networkx_graph.graph['num_of_colors'])

    plt.show(block=False)
    plt.pause(.1)
    #plt.ioff()


# Used to look at the progression of color choices.
def plot_color_pixels(fig_int, ncolors, time_series):
    plt.figure(fig_int, figsize=(5,3)); plt.clf()
    plt.title("color sequence")
    color_seq = []
    for (n, c) in time_series:
        array =  np.zeros(ncolors)
        array[c-1] = c
        color_seq.append(array)
    color_seq = np.transpose(np.asarray(color_seq))
    plt.imshow(color_seq, interpolation='nearest')
    plt.xlabel('step number', fontsize=10)
    plt.ylabel('Colors', fontsize=10)
    plt.show()



def plot(fig_int,
         rewards, mean_rewards,
         scores, mean_scores,
        num_of_strokes, mean_num_of_strokes, 
        num_of_painted_nodes, mean_num_of_painted_nodes
        ):
    display.clear_output(wait=True)
    #display.display(plt.gcf())
    fig = plt.figure(fig_int, figsize=(5, 8))
    plt.clf()

    # rewards
    plt.subplot(412)
    plt.plot(rewards)
    plt.plot(mean_rewards)
    plt.ylabel('Episodic Reward', fontsize=12)
    #plt.ylim(bottom=-2, top=2)
    plt.text(len(rewards)-1, rewards[-1], str(rewards[-1]))
    plt.text(len(mean_rewards)-1, mean_rewards[-1], str(mean_rewards[-1]))

    # # 1. scores
    # plt.subplot(311)
    # plt.plot(scores)
    # plt.plot(mean_scores)
    # plt.ylabel('Score', fontsize=12)
    # plt.ylim(ymin=0)
    # plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # 2. number of paint strokes
    plt.subplot(413)
    plt.xlabel('Number of Games')
    plt.ylabel('# of Paint Strokes', fontsize=12)
    plt.plot(num_of_strokes)
    plt.plot(mean_num_of_strokes)
    plt.ylim(ymin=0)
    plt.text(len(num_of_strokes)-1, num_of_strokes[-1], str(num_of_strokes[-1]))
    plt.text(len(mean_num_of_strokes)-1, mean_num_of_strokes[-1], str(mean_num_of_strokes[-1]))
    
    # 3. number of nodes painted
    plt.subplot(414)
    plt.xlabel('Number of Games')
    plt.ylabel('# of Nodes Painted', fontsize=12)
    plt.plot(num_of_painted_nodes)
    plt.plot(mean_num_of_painted_nodes)
    plt.ylim(ymin=0)
    plt.text(len(num_of_painted_nodes)-1, num_of_painted_nodes[-1], str(num_of_painted_nodes[-1]))
    plt.text(len(mean_num_of_painted_nodes)-1, mean_num_of_painted_nodes[-1], str(mean_num_of_painted_nodes[-1]))

    plt.show(block=False)
    plt.pause(.1)

# See the reward at each step.
def plot_step_reward(fig_int, rewards, action): 
    x = np.asarray([x for x in range(len(rewards))])
    y = np.asarray(rewards)
    plt.figure(fig_int, figsize=(5, 8))
    plt.subplot(411)
    plt.xlabel('Time Steps')
    if action == 'random':
        color = 'red'
    elif action == 'prediction':
        color = 'blue'
    plt.scatter(x, y, color=color)
    plt.plot(x, y, 'pink' )
    #plt.plot(mean_rewards)
    plt.ylabel('Step Reward', fontsize=12)
    #plt.ylim(bottom=-2, top=2)
    #plt.text(len(plot_cumulative_reward)-1, plot_cumulative_reward[-1], str(plot_cumulative_reward[-1]))
    #plt.text(len(mean_rewards)-1, mean_rewards[-1], str(mean_rewards[-1]))




""" ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
Graph Conversion Functions
!!!!!!!!!!!!!!!!!!!!!! """


# These functions are called primarily to convert between networkx functionalities 
# to Pytorch. It is organized such that all information is stored in the networkx graph,
# avoiding any call to the "game state" itself.
# The locations L for where we need to convert are:
#  L.1. "Agent.py / get_action(state, game) / step 2.B. (Exploitation) " 
#      so the GNN outputs a prediction.
#  L.2. "Model.py / train_step(state, next_state, ...) / step 1 "
#      so the GNN learns the colored graphs for both "state" and "next_state".

" ---- Convert from NetworkX to PyTorch ---- "

def convert_nodes_from_networkx_to_pytorch(networkx_graph):
    # Convert an integer representative of a node color into 
    # a 1-hot vector encoding.
    node_colors        = []
    ncolors            = networkx_graph.graph['num_of_colors']
    color_sku_template = [0] * ncolors
    for n in networkx_graph.nodes:
        color_sku = copy.deepcopy(color_sku_template)
        color_int = networkx_graph.nodes[n]['color']
        if color_int > 0: color_sku[color_int-1] = 1
        node_colors.append(color_sku)
    node_colors  = torch.tensor(np.asarray(node_colors), dtype=torch.float)
    return node_colors

def convert_edges_from_networkx_to_pytorch(networkx_graph):
    node_dict = networkx_graph.graph['node_dict'].indexed_items
    edge_list = [ [node_dict[e[0]], node_dict[e[1]]] for e in networkx_graph.edges ]
    #print('edge_list: ', edge_list)
    edge_list   = torch.tensor(np.asarray(edge_list).T, dtype=torch.long) 
    if edge_list.nelement() == 0 :
        edge_list = torch.tensor(np.array([[0], [0]]))
    return edge_list

    
" ----- Convert from PyTorch to NetworkX ------ "
def convert_node_idx_to_label(idxN, networkx_graph):
    #print('node_idx: ', idxN)
    #print('networkx_graph.nodes:', networkx_graph.nodes)
    #print('node dict: ', nx_graph_old.graph['node_dict'])
    indexed_nodes = networkx_graph.graph['node_dict'].indexed_items.items()
    node_label = [node_label for node_label, idx in indexed_nodes if idx == idxN][0]
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


" ------ Convert Graph to its Intersection Graph ------- "
def convert_to_intersection_graph(pairings: dict) -> nx.Graph:
    int_G = nx.Graph()
    # create nodes representing each color-pair which is mapped from the edges between the original graph.
    num_of_nodes = len(pairings)
    for c1 in range(1, num_of_nodes):
        for c2 in range(1 + c1, 1 + num_of_nodes):
            int_G.add_node( (c1, c2) )


