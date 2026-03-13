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
from pytorch_agent import Agent
from pytorch_game import GraphPainterAI
from gnn_model3 import GNN, FNN, QTrainer
#from pytorch_model import GCN, QTrainer
#from pytorch_model2 import GCNConv, QTrainer
import pytorch_helper


""" ------- Read the configuration data --------- """

folder_path = "/Users/joshuaroldan/painterAI"
experiment_id = 0
config_data = read_config(configfile  = f"{folder_path}/experiments/{experiment_id}/config{experiment_id}.ini")
last_game   = 500     #config_data[ 'last game']

# __Current Changes__________
# % Modify GNN: activation function to sigmoid

# __Planned Changes__________
# % L1Loss function
# % weight decay (wd ≥ -5): -6
# % Modify GNN: add a Linear Layer 
# % Change input graph into the dual graph.

def train():
    # We keep track of: rewards, scores, number of paint strokes, number of nodes used
    reward_counter          = pytorch_helper.ItemCounter()
    episodic_reward_counter = pytorch_helper.ItemCounter()
    plot_mean_episodic_rewards = [0]
    # scores
    score_counter           = pytorch_helper.ItemCounter()
    plot_mean_scores        = []
    record_score = 0
    # number of paint strokes
    paint_stroke_counter    = pytorch_helper.ItemCounter()
    plot_mean_num_of_strokes = []
    least_num_of_strokes = 10000
    # number of nodes painted
    painted_node_counter    = pytorch_helper.ItemCounter()
    plot_mean_num_of_painted_nodes = []
    least_num_of_nodes_used = 10000

    # Initiate the game and agent.
    game  = GraphPainterAI()
    agent = Agent(game)
    glia  = pytorch_helper.GliaCell(); agent.attach_a_glia_cell(glia)

    # Game specs.
    #print(' Number of colors: ',game.ncolors); print(' Number of nodes : ',game.num_of_nodes); print(' Number of edges : ',game.num_of_edges)
    
    plt.ion()

    """ -------------------------------------------- "
                Steps to Play the Game 
    " ---------------------------------------------- "
      1. A. Get old game state. , B. Visualize.
       # convert networkx to pytorch.
      2. 'get_action()': Get 'recommended' node and color.
       # convert pytorch to networkx.
      3. 'play_step()': Paint 'recommended' node and color.
      4. A. Get new game state. , B. Visualize.
       # convert networkx to pytorch.
      5. Train short memory.
      6. Remember.
      (Repeat 1.- 6.) unless...
      7. (If done), collect game statistics.
    " ---------------------------------------------- """

    agent.glia.notes['graph'] = ''

    while True:
        # 1.A. Get old game state.
        print('\nGetting (old) game state...') ; agent.glia.notes['graph'] = 'old'
        nx_state_old = agent.get_state(game)
        pairs_vector = nx_state_old[ 'pairs vector' ]
        nx_graph_old = nx_state_old[ 'painting and inner frame' ] #config_data['state'] ]
        # 1.B. Visualize (old) graph state.
        fig_int= 1; pytorch_helper.visualize_colored_graph(fig_int, networkx_graph= nx_state_old[ 'painting and inner frame' ] ) #nx_graph_old)

           # Convert graph to intersection graph.
        old_pairings = nx_state_old[ 'pairings' ]
        #intersection_graph = convert_to_intersection_graph(old_pairings)
        

           # Convert NetworkX graph to PyTorch.
        py_node_colors = pytorch_helper.convert_nodes_from_networkx_to_pytorch(networkx_graph=nx_graph_old)
        py_edge_list   = pytorch_helper.convert_edges_from_networkx_to_pytorch(networkx_graph=nx_graph_old)
        py_graph_old   = Data(x=py_node_colors, edge_index=py_edge_list)
        py_graph_old.validate(raise_on_error=True)
        agent.glia.approve_graph_construction( nx_graph_old, py_graph_old)
        #print(' PyTorch Graph of (old) state: ', py_graph_old)

        # 2. Get the model's recommended node and color.
        print('\nGetting next node and color via ...')
        idxN, final_color_sku = agent.get_action(pairs_vector, nx_graph_old, py_graph_old)
        agent.glia.approve_action(idxN, nx_graph_old, py_graph_old)

           # Convert PyTorch objects to NetworkX.
        node_label = pytorch_helper.convert_node_idx_to_label(idxN, nx_graph_old)
        color_int  = pytorch_helper.convert_color_sku_to_int(final_color_sku)
        #assert node_label not in game.inner_frame.nodes, f" idxN {idxN}/ node {node_label} is not a boundary {game.inner_frame.nodes}."

        # 3. Paint 'recommended' node and color.
        print('\nPlaying action... ')
        done, reward, score = game.play_step(node_label, color_int)
        reward_counter.add_items(reward)

           # Plot some aspects about the decision.
        # plot step rewards
        fig_int= 2; pytorch_helper.plot_step_reward(fig_int, rewards= reward_counter.time_series, action=agent.glia.notes['action'])
        # Pixel plot of color choices.
        fig_int= 4; pytorch_helper.plot_color_pixels(fig_int, ncolors= game.ncolors, time_series= game.sequence_of_paint_strokes)

        # 4.A. Get new game state.
        print('\nGetting (new) game state...')
        nx_state_new = agent.get_state(game)
        nx_graph_new = nx_state_new[ config_data['state'] ] #; nx_graph_new = nx_state_new['entire graph']
        # 4.B. Visualize (new) colored graph .
        fig_int= 1; pytorch_helper.visualize_colored_graph(fig_int, networkx_graph= nx_state_new[ 'painting and inner frame' ] ) #nx_graph_new)

             # Convert NetworkX graph to PyTorch.
        py_node_colors = pytorch_helper.convert_nodes_from_networkx_to_pytorch(networkx_graph=nx_graph_new)
        py_edge_list   = pytorch_helper.convert_edges_from_networkx_to_pytorch(networkx_graph=nx_graph_new)
        py_graph_new   = Data(x=py_node_colors, edge_index=py_edge_list)
        py_graph_new.validate(raise_on_error=True)
        agent.glia.approve_graph_construction(nx_graph_new, py_graph_new)
        #print(' PyTorch Graph of (new) state: ', py_graph_new)

        # 5. Train short memory.
        print('\nTraining (short) term memory...')
        #print('old state x shape',py_graph_old.x.shape)
        #print('old state x',py_graph _old.x) ; print('\nnew state x',py_graph_new.x) , print('\nidxN',idxN)
        agent.train_short_memory(py_graph_old, py_graph_new, idxN, final_color_sku, reward, done)

        # 6. Remember.
        #for dd, decision in enumerate(agent.memory): print('memory ', str(dd), decision)
        agent.remember(py_graph_old, py_graph_new, idxN, final_color_sku, reward, done)

        # 7. (If done), collect game statistics.
        if done: # done is True when the game is over
            # some statistics of the game played.
            num_of_strokes       = len(game.sequence_of_paint_strokes)
            num_of_painted_nodes = len(game.painting)

            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            print('\nTraining (long) term memory...')
            agent.train_long_memory()

            # Decide if to save model.
            save_model = False
            if score > record_score:
                record_score = score
                save_model   = True
            if num_of_strokes < least_num_of_strokes:
                least_num_of_strokes = num_of_strokes
                save_model   = True
            if num_of_painted_nodes < least_num_of_nodes_used:
                least_num_of_nodes_used = num_of_painted_nodes
                save_model   = True
            if save_model == True:
                pass
                #agent.model.save()

            #print('Game', agent.n_games, 'Score', reward, 'Record:', record)
            # print('Game:', agent.n_games, 'Score:', score, 'Stroke Count:', num_of_strokes, 'Cells Painted:', num_of_painted_nodes,'\n'\
            #        'Record score:', record_score, 'Record strokes:', record_strokes, 'Record cells painted:', record_cells_used)
            print('\n\nNew Game.')

            # plot episodic rewards
            episodic_reward_counter.add_items( reward_counter.count )
            plot_mean_episodic_rewards.append( episodic_reward_counter.average_count() )
            # scores
            score_counter.add_items( score ) ;  assert agent.n_games == len(score_counter.time_series), "check n_game or score counter"
            plot_mean_scores.append( score_counter.average_count() )
            # number of paint strokes
            paint_stroke_counter.add_items( num_of_strokes ) ;  assert agent.n_games == len(paint_stroke_counter.time_series), "check n_game or paint stroke counter"
            plot_mean_num_of_strokes.append( paint_stroke_counter.average_count() )
            # number of nodes painted
            painted_node_counter.add_items( num_of_painted_nodes ) ;  assert agent.n_games == len(painted_node_counter.time_series), "check n_game or painted node counter"
            plot_mean_num_of_painted_nodes.append( painted_node_counter.average_count() )
            # Plot statistics.
            fig_int = 2; pytorch_helper.plot(fig_int,
                        episodic_reward_counter.time_series, plot_mean_episodic_rewards,
                        score_counter.time_series, plot_mean_scores, 
                        paint_stroke_counter.time_series, plot_mean_num_of_strokes,
                        painted_node_counter.time_series, plot_mean_num_of_painted_nodes
                        )
            
            # rewards
            plt.figure(2); plt.subplot(411); plt.cla()
            reward_counter.reset_count()

        if agent.n_games == last_game:
           plt.show(block=True)
           break


if __name__ == '__main__':
    train()