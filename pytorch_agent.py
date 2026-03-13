from read_config import read_config
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque # deque is a list optimized for removing and adding items
import time
import copy
import seaborn as sns

import helper as h
from pytorch_game import GraphPainterAI
from gnn_model3 import GNN, FNN, QTrainer
#from pytorch_model import GCN, QTrainer
#from pytorch_model2 import GCNConv, QTrainer
import pytorch_helper


""" ------- Read the configuration data --------- """

folder_path = "/Users/joshuaroldan/painterAI"
experiment_id = 0
config_data = read_config(configfile  = f"{folder_path}/experiments/{experiment_id}/config{experiment_id}.ini")

MAX_MEMORY = 100_000 # the size of the deque
BATCH_SIZE = 1000    # mini batch size
# LR = 0.01 was pretty good for 4 colors.
LR         = 0.01   # learning rate # config_data['learning rate'] #

class Agent:
    def __init__(self, game):
        self.n_games = 0
        self.epsilon = 0        # randomness 
        self.gamma   = 0.08     # config_data['gamma'] # discount rate 
        self.memory  = deque(maxlen=MAX_MEMORY)        # popleft() if MAX_MEMORY is reached. The agent stores the last 100_000 moves in memory
        self.model   = GNN(input_size  = game.ncolors,
                           hidden_size = 64, #32           #64 config_data['hidden size'], 
                           output_size = game.ncolors)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def attach_a_glia_cell(self, glia):
        self.glia = glia
    
    ### class methods ###

    # 1. Get game state. This method takes an argument, 'game', which is of the class GraphPainterAI in game.py.
    #    Output ('state') includes NetworkX graph.
    def get_state(self, game):
        # 1.1. First, get the current pairings. 
        pairings = game.pairings #; print(' color pairings: ', pairings)
        # 1.2. Returns a list of integers indicating how many different colors a given color is adjacent to.
        pairs_vector = [len(pairings[color]) for color in game.vcolors] #; print(' pairs vector: ', pairs_vector)
        # 1.3. Return a list of bool indicating if color C has completed all of its pairings. # this is a list with boolean values.
        completed_pairs_vector = [int(len(pairings[color]) == game.ncolors - 1) for color in game.vcolors] #; print(' completed pairs vector: ', completed_pairs_vector)
        # 1.4. NetworkX colored graphs.
        # 1.4.A. Entire graph.
        entire_graph_state       = game.colored_graph
        # 1.4.B. Painting and inner frame.
        painting_and_inner_frame = game.colored_graph.subgraph( set(game.painting.nodes).union( game.inner_frame.nodes ) )
        # 1.4.C. Inner frame.
        indexer      = game.colored_graph.graph['node_dict']
        idxBN_sample = indexer.sample_indexed_items(sample_keys= game.inner_frame.nodes)
        #idx_sample  = indexer.reindex_items(idxBN_sample)
        game.inner_frame.graph['node_dict'] = copy.deepcopy(idxBN_sample)
        inner_frame = game.inner_frame
        # 1.5. Specify what constitutes the game state.
        state = {'pairings'                : pairings,
                 'pairs vector'            : pairs_vector,
                 'entire graph'            : entire_graph_state,
                 'painting and inner frame': painting_and_inner_frame,
                 'inner frame'             : inner_frame,
                 }
        return state
    
    # 'get_action()' :: method changes "final_move" to include any painted cell as the starting position.
    #    Input includes a NetworkX graph and PyTorch version.
    #    Output is two vectors (skus) to specify the next node and color.
    def get_action(self, pairs_vector, nx_graph_old, py_graph_old):
        # 1. Determine process for selecting next move:
        #    - Process 1: (Exploration ) Random selection process.
        #    - Process 2: (Exploitation) Use the neural network to generate a selection.
        # 2.A. (Exploration):
        #    Consider two methods for exploration, i.e. randomly selecting a node.
        #    - Method 1: Select any node and any color.
        #                A. Randomly select a node.
        #                B. Randomly select a color.
        #    - Method 2: Select any neighbor of a painted node and any color.
        # 2.B. (Exploitation): 
        #    Get node and color choice from Q-network.
        #    Input 'state' to output 'predicted Q vals' with the Q-network,
        #   2.B.1. Convert state's colored graph into a torch_geometric.data.Data.
        #     2.B.1.1. Convert graph nodes.
        #     2.B.1.2. Convert graph edges.
        #     2.B.1.3. Create PyTorch graph.
        #   2.B.2. Get the prediction of our GNN model based on the current graph state.
        #   2.B.3. Define MAX function and get the index of the maximum value of the prediction.
        #    - Method 1: Find the maximum value for the entire matrix.
        #    - Method 2: Compute norm of each node feature vector, then select the maximum argument.
        #                A. Chose the node with the maximum norm of their feature vector.
        #                B. Chose the color with the maximum Qval.
        #                C. Get index based on MAX value.

        # Trade-off Exploration / Exploitation
        final_color_sku = [0] * nx_graph_old.graph['num_of_colors']
        self.epsilon    =  200 - self.n_games # deafult: 100 - self.n_games

        if random.randint(0, 150) < self.epsilon: # if the random number is less than epsilon, then choose a random action
            print(' ... randomness:'); action = 'random'
            # 2.A. (Exploration): 
            #   Method 1: Select any node and any color.
            #   Method 2: Select any neighbor of a painted node and any color.
            # ** Method 3: Select a node from the graph boundary.
            # Method 3.A. Get the boundary nodes' indices.
            idxN = self.choose_boundary_node_from_a_uniform_distribution(nx_graph_old)
            idxC = self.choose_color_from_a_weighted_distribution(pairs_vector)
            final_color_sku[idxC] = 1

            # Display the Qmat for visual continuity.
            np_prediction  = self.model(py_graph_old.x, py_graph_old.edge_index).detach().numpy()
            boundary_indices = nx_graph_old.graph['node_dict'].sample
            np_prediction  = self.glia.filter_out_graph_interior_from_prediction(np_prediction, boundary_indices)
            plt.figure(3, figsize=(3.5, 3)); plt.clf()
            sns.heatmap(np_prediction, cmap='gray_r')

            
        else: 
            print(' ... GNN prediction:'); action = 'prediction'
            # 2.B. (Exploitation): Get node and color choice from Q-network.
            # 2.B.2. Get the prediction of our GNN model based on the current graph state.
            print(' Inputting (old) game state into GNN ...' )
            print('   nodes: ', py_graph_old.x.shape         )
            print('   edges: ', py_graph_old.edge_index.shape)
            # Note: The prediction is an n x c matrix where n is number of nodes, and c is the number of colors we are playing with.
            prediction = self.model(py_graph_old.x, py_graph_old.edge_index).detach().numpy()
            print(" GNN predicts: ", prediction.shape)#, '\n', prediction)

            # 2.B.3. Define MAX function and get the index of the maximum value of the prediction.
            #  Method 1: Find the maximum value for the entire matrix.
            #  Method 2: Compute norm of each node feature vector, then select the maximum argument.
            print(' Getting info for the training step ...')
            method = 1
            if method == 1:
                # Get only the nodes part of the graph boundary.
                boundary_index = nx_graph_old.graph['node_dict'].sample
                prediction     = self.glia.filter_out_graph_interior_from_prediction( prediction, boundary_index)
                # Find the maximum Q value, and select a node-color pair from the preimage.
                Qmax_preimage   = np.where(prediction==prediction.max())
                idxBNs,max_colors= Qmax_preimage ; print('  idxBNs',idxBNs,'max_colors',max_colors)
                pair_idx        = random.randrange(0, len(idxBNs) )
                print(f' randomly picked node-color pair {pair_idx}/{len(idxBNs)} from preimage.')
                print( '  max idxBN : ', idxBNs[pair_idx]) ; print('  max color: ', max_colors[pair_idx])
                idxN  = boundary_index[ idxBNs[pair_idx] ] ; print(' idxN: ', idxN)
                final_color_sku[max_colors[pair_idx]] = 1

            if method == 2:
                norm = np.linalg.norm(prediction, axis=1) #; print('norm axis 1: ', norm.shape, norm)
                # Method 2.A. Chose the node with the maximum norm of their feature vector.
                max_norm_node  = np.argmax(norm)
                print(' max_norm_node: ', max_norm_node)
                # Method 2.B. Chose the color with the maximum Qval.
                print( 'max norm node Qval vector: ', (prediction)[max_norm_node,:])
                max_color = np.argmax((prediction)[max_norm_node,:])
                print(' max_color: ', max_color)
                # Method 2.C. Get index based on MAX value.
                final_node_sku[max_norm_node] = 1 # set the final move to 1 at the index of the maximum value of the prediction
                final_color_sku[max_color]    = 1 # as the game progresses, we will perform less and less random moves and more 
                                                    # and more moves based on the prediction of our model
                                                    # as the game progresses, we will perform less and less random cells and more and more cells based on the prediction of our model
                print('final_color_sku: ', final_color_sku)
        self.glia.notes['action'] = action
        return idxN, final_color_sku

    " ----- 'Get Action': Methods for Random Selection ------ "

    def choose_boundary_node_from_a_uniform_distribution(self, nx_graph_old):
        boundary_indices  = nx_graph_old.graph['node_dict'].sample
        # print(' # of boundary nodes: ', len(boundary_indices.items()))
        # print('       boundary idxs: ', boundary_indices.items() )
        # Method 3.B. Randomly select a node.
        idxBN = random.randrange(0, len(boundary_indices.items())); print(' boundary idx: ', idxBN)
        idxN  = boundary_indices[idxBN]                           ; print(' node idx: ', idxN)
        return idxN
    
    def choose_color_from_a_weighted_distribution(self, pairs_vector:list):
        pairs_vector = np.asarray(pairs_vector)
        pairs_vector = 1 - pairs_vector
        minimum      = pairs_vector.min()
        preimage     = np.where(pairs_vector == minimum)
        idxC         = random.randrange( 0, len(pairs_vector) )
        return idxC
    
    " -------------------------------------------------------- "



    # Decision is a tuple containing the action to move and the color_choice made by the agent
    def remember(self, state, next_state, idxN, final_color_sku, reward, done): 
        self.memory.append((state, next_state, idxN, final_color_sku, reward, done))

    def train_short_memory(self, next_state, state, idxN, final_color_sku, reward, done):
        self.trainer.train_step(state, next_state, idxN, final_color_sku, reward, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # if the memory is full, then sample a mini batch of size BATCH_SIZE from the memory
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else: # if the memory is not full yet, then sample the whole memory
            mini_sample = self.memory
        # use the trainer to train the model. First 'unpack' the mini_sample into its components
        #states, final_node_skus, final_color_skus, rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, final_node_skus, final_color_skus, rewards, next_states, dones)
        # can also be done with a for loop:
        for state, next_state, idxN, final_color_sku, reward, done in mini_sample:
            self.trainer.train_step(state, next_state, idxN, final_color_sku, reward, done)