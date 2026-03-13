from read_config import read_config
import pytorch_helper

#import pygame 
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import matplotlib.pyplot as plt
#from scipy.special import comb
import copy
import time

folder_path = "/Users/joshuaroldan/painterAI"
experiment_id = 0
config_data = read_config(configfile  = f"{folder_path}/experiments/{experiment_id}/config{experiment_id}.ini")


# Print out commentary and results during game play?
verbose = False 
SPEED   = 2000

""" ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
Initiating the underlying colored graph.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! """

# Q: What is the underyling graph topology we play on?
#grid_dimensions  = [10, 10]
UNDERLYING_GRAPH = nx.grid_graph(dim=[config_data['grid dimension 1'], config_data['grid dimension 2']], periodic=False)
NUM_OF_NODES     = UNDERLYING_GRAPH.number_of_nodes()
NUM_OF_EDGES     = UNDERLYING_GRAPH.number_of_edges()
# Color all the nodes black.
for idx, n in enumerate(UNDERLYING_GRAPH.nodes):
    UNDERLYING_GRAPH.nodes[n]['color'] = 0 ; #print('n: ', n)

# Save the underlying graph.
nx.write_graphml(G=UNDERLYING_GRAPH,
                path='/Users/joshuaroldan/data/painterAI/UNDERLYING_GRAPH.graphml')

# The node/ edge labels are standardized to be integers.
node_dict = {}
for idx, v in enumerate(UNDERLYING_GRAPH.nodes):
    node_dict[v] = idx
edge_dict = {}
for e in UNDERLYING_GRAPH.edges:
    #print('underlying graph e: ', e)
    v0_idx = node_dict[e[0]]
    v1_idx = node_dict[e[1]]
    edge_dict[(e[0], e[1])] = [v0_idx, v1_idx]
    edge_dict[(e[1], e[0])] = [v1_idx, v0_idx]
## Playing with Networkx Methods ##
for n, neighbors in UNDERLYING_GRAPH.adj.items():
    #print('n: ', n, 'neighbors: ', neighbors)
    node_color = UNDERLYING_GRAPH.nodes[n]['color']
    for neighbor_name, edge_attr in neighbors.items(): 
        #print('neighbor node name: ', neighbor_name):
        neighbor_attrs = UNDERLYING_GRAPH.nodes[neighbor_name]; #print('neighbor_node_attrs: ', neighbor_attrs)
        neighbor_color = UNDERLYING_GRAPH.nodes[neighbor_name]['color']; #print('neighbor color: ', neighbor_color)


""" ------------------------------------------------------------- """
""" --- Implement the coloring problem on any graph topology  --- """
""" ------------------------------------------------------------- """


# Q: How many numbers are we playing with?
NUM_OF_COLORS = 4 #config_data['number of colors']

class GraphPainterAI: # Inherits from the class 'object'. This is the same as 'class GraphPainterAI(object):'
    ### class initializer ###
    def __init__(self, num_of_nodes= NUM_OF_NODES, num_of_edges= NUM_OF_EDGES, ncolors = NUM_OF_COLORS):
        self.num_of_nodes = num_of_nodes
        self.num_of_edges = num_of_edges
        self.ncolors = ncolors
        self.vcolors = [c for c in range(1, 1+self.ncolors)] # this is a list of the colors that the agent can choose from, as integers
        self.total_color_permutations = self.ncolors*(self.ncolors-1)
        self.pairings = {} # this is a dictionary (keys, values).dtype = (int, int) which tracks which colors have been paired.
        self.reset()


    ### class methods ###
    # 1. reset. This initializes the game state.
    def reset(self): # it does not take any arguments, since it is intended to act only when the condition of game-over or win is met.
        
        print('\nReseting graph ...')
        # Reset the colored graph to be the BLACK underlying graph.
        self.colored_graph = copy.deepcopy(UNDERLYING_GRAPH)
        # Attribute to the graph the number of colors to play the game with.
        self.colored_graph.graph['num_of_colors'] = self.ncolors
        # Attribute to the graph a list of nodes painted.
        self.colored_graph.graph['painted_nodes'] = []; # print('node data: ', self.colored_graph.nodes.data()); print('edge data: ', self.colored_graph.edges.data())
        self.colored_graph.graph['node_dict'    ] = {} #node_dict
        self.colored_graph.graph['edge_dict'    ] = {} #edge_dict

        # Initialize the dictionary that tracks the number of color-pairs.
        self.pairings = {color: [] for color in self.vcolors} 
        # Set the induced subgraph with colored nodes.
        self.painting = self.colored_graph.subgraph(self.colored_graph.graph['painted_nodes']).copy()  
        # Record the painting process. (may includes repeated nodes.)
        self.sequence_of_paint_strokes = [] 

        # Start/ Reset the painter.
        self.painter  = (config_data['grid dimension 1']/2, config_data['grid dimension 2']/2) # Start the painter in the middle of the graph.
        color_int     =  1     # Set the first node color as color 1 (aka BLUE)
        self._paint_node(color_int) # Paint node.
        self._update_painting()  # Update the painting to reflect the newly painted node.
        print('Start painter: ', self.painter, 'color: ', self.colored_graph.nodes[self.painter]['color'])
        
        # Start / Reset the new graph node indexer.
        self.indexer = pytorch_helper.Indexer()
        self.colored_graph.graph['node_dict'] = self.indexer
        self.indexer.index_new_item(self.painter)

        # The (inner) frame / boundary surrounding the Painting. Outer frame is the boundary of the lattice.
        # This gives us the graph of neighbors.
        self.inner_frame = self.colored_graph.subgraph( set(self.colored_graph.neighbors(self.painter)) ).copy() 
        for node_label in self.inner_frame.nodes:  self.indexer.check_index(node_label)

        # Start / Reset the game pairs/ score.
        self.pairs_per_color = np.zeros(self.ncolors)
        self.old_total_pairs = 0
        self.score           = 0


    # 2) play_step. This is the main method of the class. It takes an argument, 'move', which is the move taken by the agent.
    def play_step(self, node_label, color_int):
        # 2.2) play step:
        #    2.2.1) move painter
        #    2.2.2) paint node
        #    2.2.3) update painting
        #    2.2.4) update pairing
        # 2.3) check if step causes game over:
        #    2.3.2) won the game
        #    2.3.3) graph is full
        # 2.4) reward/punish 'strategy':
        #    2.4.1) reward a new pair/ punish lost pair
        #    2.4.2) reward/punish # of paint strokes # ????????
        #    2.4.3) reward/punish # of used nodes in painting  # ??????
        #    2.4.4) punish pattern (ex. lines)
        #    2.4.5) reward for different colored neighbors.



        old_number_of_nodes = self.painting.number_of_nodes()
        # 2.2) Play step
        # 2.2.1) move painter   # Move the painter to the assigned node.
        self._move_to_node(node_label) # this method is defined below. (changes 'self.painter' variable).
        # 2.2.2) paint node
        self._paint_node(color_int)
        # 2.2.3) update painting
        self._update_painting()
        # 2.2.4) update pairings. Note: it suffices to merely inspect the 'painting' and not the entire 'colored_graph'.
        self._update_pairings()
        # 2.2.5) update inner frame.
        self._update_inner_frame()
        # 2.2.6) update node dictionary.
        self._update_node_dictionary()
        new_number_of_nodes = self.painting.number_of_nodes()

        # Visualize the painting.
        painting_and_inner_frame = self.colored_graph.subgraph( set(self.painting.nodes).union( self.inner_frame.nodes ) )
        fig_int= 1; pytorch_helper.visualize_colored_graph(fig_int, networkx_graph=painting_and_inner_frame )
        # Pixel plot of color choices.
        fig_int= 4; pytorch_helper.plot_color_pixels(fig_int, ncolors= self.ncolors, time_series= self.sequence_of_paint_strokes)


        # Assign Rewards/Punishments.
        reward = 0
        # 2.3) Check if game is over:
        game_over = False 
        # 2.3.2) won the game 
            # The game is won once all of the pairings have been completed.
            # This means that the length of the list of pairings of each color is equal to the number of colors minus one.
            # In that case we will give a large positive reward.
        if self.game_won():
            print('All colors paired. You win!')
            game_over = True
            reward += 100


        " ---- Reward/Punish based on Global properties ------------------------------------ "
        number_of_paint_strokes = len(self.sequence_of_paint_strokes)
        # Start at the second move.
        if number_of_paint_strokes > 1:

            # Reward choosing nodes with less pairs.
            if self.pairs_per_color[color_int -1] < self.pairs_per_color.max():
                reward += 3

            # 2.3.3) Exceeds upperbound.
            if self.ncolors > 3:
                upperbound = (self.ncolors - 1)*(self.ncolors - 2) * 2
            elif self.ncolors == 3:
                upperbound =  4
            # Boolean valued.
            upperbound_passed = (len(self.colored_graph.graph['painted_nodes']) > upperbound); print(f' Bound passed?  : {upperbound_passed}')    
            if upperbound_passed:
                game_over = True
                reward += 0
                #reward += -0.01 * (len(self.colored_graph.graph['painted_nodes']) - upperbound_passed)

            # 2.4) Reward/ Punish 'strategy':
            # 2.4.0) Punish each move (This is to incentivize faster convergence.)
            reward += -1.5
            # 2.4.1) reward a new pair/ punish losing a pair.
            old_pairs_per_color = copy.deepcopy(self.pairs_per_color)
            for c in self.vcolors: # loops through the colors. 
                self.pairs_per_color[c-1] = len(self.pairings[c])  # list of paired colors for a given color c # Have to adjust the 1-based color index to 0.
            # Reward for having minimum pairs of any color non-zero. (Should help with increasing the expected reward)
            if self.pairs_per_color.min() > 0:
                reward += 1 
            # Compare color-pair counts.
            diff = np.subtract( self.pairs_per_color, old_pairs_per_color ) #; print(' diff: ', diff); print(' diff < 0:' , (diff < 0)); print(' np.any((diff < 0)): ', np.any((diff < 0)))
            colors_paired = np.where(diff>0)
            # Boolean values.
            pairs_added = np.any((diff > 0)) ; print(f' Pairs added?   : {pairs_added}')
            pairs_lost  = np.any((diff < 0)) #; print(f' Pairs lost?    : {pairs_lost}')
            constant    = np.all((diff== 0)) #; print(f' Pairs constant?: {constant}')
            if pairs_added and not pairs_lost:
                reward += 8 * sum((diff > 0))
                print('diff ', diff)
                print(' Number of colors with at least one new pair:', sum((diff > 0)))
            # Reward first pairs.
            if 0 in old_pairs_per_color[colors_paired]:
                reward += 12
            elif pairs_lost:
                pass
                reward += -1 * sum((diff < 0))
            # Besides the first move, assumes every move should add a pair.
            elif constant: 
                # Punish choosing nodes with most pairs.
                if self.pairs_per_color[color_int -1] == self.pairs_per_color.max():
                    # Punish if a color which already has all its pairs is chosen.
                    if self.pairs_per_color.max() == self.ncolors-1:
                        reward += -20
                    # Punish for not selecting a node less paired as there is a higher probability of it getting paired.
                    else:
                        reward += -10
                # Punish for not making a pair.
                else:
                    reward += -1

        " ---- Reward/Punish based on Local properties ------------------------------------ "
        number_of_paint_strokes = len(self.sequence_of_paint_strokes)
        # Start at the second move.
        if number_of_paint_strokes > 1:                        
            if not pairs_added:
                # 2.4.5. Reward for different colored neighbors. Punish if all neighbor colors are the same.          
                for neighbor in list(self.painting.neighbors(self.painter)):
                    neighbor_color = self.painting.nodes[neighbor]['color']
                    # Discourage existing pairs.
                    if neighbor_color in self.pairings[color_int]:
                        reward += -2
                    elif color_int == neighbor_color:
                        reward += -10

        self.score = sum(self.pairs_per_color)/self.total_color_permutations # A normalized version of the score.
        if verbose == True:
            pass
            print(" prior number of pairs: "        , self.old_total_pairs)
            print(" current number of pairs: "      , self.pairs_per_color)
            print(" self.total_color_permutations: ", self.total_color_permutations)
            print(" score: "                        , self.score)

        print(' total reward: ', reward)


         # 2.5. return game over and score
        return game_over, reward, self.score


    """ ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
    Dependent functions 
    !!!!!!!!!!!!!!! """

    # Painter gets moved to a new node via "_move_to_node()", then "_paint_node()" colors the new node.
    def _move_to_node(self, node_label):
        # In the 'nx.grid" graph, the node labels are named (i,j) where i is the row and j is the columnn. 
        self.painter = node_label

    # Color the Painter's node.
    def _paint_node(self, color_int):
        self.colored_graph.nodes[self.painter]['color'] = color_int
        self.sequence_of_paint_strokes.append((self.painter, color_int))
        self.colored_graph.graph['painted_nodes'] = set([node for (node, color) in self.sequence_of_paint_strokes])

    # Update the painting.
    def _update_painting(self):
        self.painting = self.colored_graph.subgraph(self.colored_graph.graph['painted_nodes']).copy()
    
    # Update the pairings based on the painting.
    def _update_pairings(self):
        # Reset the pairings and recalculate the current pairs.
        self.pairings = {color: [] for color in self.vcolors} 
        # Check all the nodes in the painting and update the pairings dictionary by checking the neighboring node colors.
        for n, neighbors in self.painting.adj.items():
            #print('n: ', n, 'neighbors: ', neighbors)
            node_color = self.painting.nodes[n]['color']
            for neighbor_name, edge_attr in neighbors.items():
                #print('neighbor node name: ', neighbor_name)
                neighbor_attrs = self.painting.nodes[neighbor_name]
                #print('neighbor_node_attrs: ', neighbor_attrs)
                neighbor_color = self.painting.nodes[neighbor_name]['color']
                #print('neighbor color: ', neighbor_color)
                if (node_color != 0) and (neighbor_color != 0):
                    if (neighbor_color != node_color) and (neighbor_color not in self.pairings[node_color]):
                        self.pairings[node_color].append(neighbor_color)

    def _update_inner_frame(self):
        # remove new node from inner frame.
        if self.painter in self.inner_frame.nodes:
            self.inner_frame.remove_node(self.painter)
        # get the neighbors of this node.
        for neighbor in self.colored_graph.neighbors(self.painter):
            # if a neighbor is not painted, then add it to the frame.
            if neighbor not in set(self.painting.nodes):
                self.inner_frame.add_node(neighbor)
        # get the edges connecting the frame.
        self.inner_frame = self.colored_graph.subgraph( self.inner_frame.nodes ).copy()

    def _update_node_dictionary(self):
        for node_label in self.inner_frame.nodes:  
            self.indexer.check_index(node_label) 

    def graph_full(self): # If the board is full, then the game is over.
        # check if the board is full. This is simple: if the length of the list painting is equal to the number of cells in the board, then the board is full.
        if len(self.painting.keys()) == (self.w//BLOCK_SIZE)*(self.h//BLOCK_SIZE):
            return True
        
    def game_won(self): # If the game is won, then the game is over.
        # check if the game is won. This is simple: if the length of the list of pairings of each color is equal to the number of colors minus one, then the game is won.
        # vector of booleans. Each boolean is True if the length of the list of pairings of the corresponding color is equal to the number of colors minus one.
        won = [len(self.pairings[color]) == self.ncolors - 1 for color in range(1, self.ncolors)]
        # if all the elements of the vector are True, then the game is won.
        if all(won):
            return True
