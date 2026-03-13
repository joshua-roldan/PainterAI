import torch
from torch_geometric.data import Data
from torch.nn import Linear, Parameter
from torch_geometric.nn import GCNConv, MessagePassing, SimpleConv
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import time

import pytorch_helper

verbose = False



class FNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        torch.manual_seed(12345)
        self.linear1 = Linear(input_size , hidden_size)
        self.linear2 = Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x
    

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_size , hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.sigmoid()
        x = self.conv2(x, edge_index)
        x = x.sigmoid()
        x = self.conv3(x, edge_index)
        return x
    
class simpleGCN(torch.nn.Module):
    def __init__(self):
        super(simpleGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SimpleConv(aggr="sum", combine_root="sum")
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x



class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr    = lr
        self.weight_decay= 5*lr*10**(-5) #-5
        self.gamma = gamma # parameter for how much agent cares about rewards in immediate versus distant future.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay) # The way the adam optimizer works is that it takes the parameters of the model and updates them based on the loss function
        self.scheduler = CyclicLR(self.optimizer, 
                          max_lr = self.lr*(10**1), #10**1
                          base_lr = self.lr*(10**-2),
                          step_size_up = 200,
                          mode = "triangular2",
                          cycle_momentum=False)
        self.criterion = torch.nn.MSELoss() # mean squared error: (y - y_pred)^2
        #self.criterion = torch.nn.L1Loss()  # absolute value

    # 'train_step()' :: This method takes the colored graphs and trains the network to predict the right Qvals.
    def train_step(self, state, next_state, idxN, color_sku, reward, done):
        # 0. Suppose a graph time series...
        # 1. Convert "node_sku", "color_sku"
        #    b/c 
        # 2. Output model's prediction given (PyTorch) "state",
        #    i.e. predicted Q values with current state. 
        #                  Qvals_current = model(state)
        #    Q values are the values of each color for the current state.
        #    Note that ('pred'), and thus ('target'), is a nxm matrix
        #    where n:= # of nodes, m:= # of colors.
        #    2.1. Input ('state') and output the prediction matrix ('pred').
        #    2.2. Create a copy ('target') from ('pred').
        # 3. Update Qvals (with Bellman Eqn): 
        #         Recall: (Bellman Eqn)  Recursively defines current Qval in terms of current reward and next Qval.
        #                  Q_new = current_reward + discount * max(next predicted Q values)
        #    3.1. Get the current reward ('current_reward'). 
        #                  Q_new = CURRENT_REWARD + discount * max(next predicted Q values)
        #      Or, Get the preliminary Qvals for the node ('Q_newN') and color ('Q_newC'). 
        #    3.2. (If game is not over), then calculate the Q_new values with the Bellman Eqn.
        #         3.2.1. Input the ('next_state') into the GNN, and output the next predicted Q values ('next_pred').
        #                  Q_new = current_reward + discount * max(NEXT PREDICTED Q VALUES)
        #         3.2.2. Define MAX function and get the max values.
        #                  Q_new = current_reward + discount * MAX(next predicted Q values)
        #                There are two methods for max node ('max_node') and max color ('max_color'):
        #                 - Method 1: Take the max Qval of the entire matrix.
        #                 - Method 2: A. Take the max norm among node feature vectors, then
        #                             B. Take the max value for the color.
        #         3.2.3. Compute the new Qval(s) given the Bellman Eqn and the MAX value(s).
        #                  Q_NEW = current_reward + discount * max(next predicted Q values)
        #         3.2.4. Update the Qval(s) in the ('target') matrix:
        #                ('target')[row idx, col idx] = new Qval ('Q_new').
        #                s.t. row idx = ('node') idx , col idx = ('color') idx
        #    (Repeat for ?)
        # 4. Compute loss and backpropagate.
        #                 loss = ( Q_new - Q_current )^2
        #         
        # 1. Convert ('node_sku'), ('color_sku'), ('reward').
        print(' node idx: ', idxN)
        idxC       = pytorch_helper.convert_color_sku_to_int(color_sku)
        print(' color idx: ', idxC)
        reward     = torch.tensor(reward, dtype=torch.float)
        reward     = torch.unsqueeze(reward, 0)
        done       = (done, )
 
        # 2. Output model's prediction given "state", i.e. predicted Q values with current state. 
        #    Q values are the values of each color for the current state.
        #    Note that ('Q_mat'), and thus ('target'), is a nxm matrix where n:= # of nodes, m:= # of colors.
        #  2.1 Input ('state') and output the prediction matrix ('Q_mat').
        Q_mat  = self.model(state.x, state.edge_index)
        assert idxN in list(range(Q_mat.shape[0])), f"{idxN} not in index range {Q_mat.shape[0]}"
        #  2.2 Create a copy ('target') from ('Q_mat').
        target = Q_mat.clone()

        # 3. Update Qvals (with Bellman Eqn): 
        #    Recall: (Bellman Eqn)  Recursively defines current Qval in terms of current reward and next Qval.
        #    Q_new = r_current + discount * max(next predicted Q values)
        for paint_stroke_id in range(len(done)): #len(done) equals the number of steps played over all games. ( len(done)= total number of strokes over all games )
           # 3.1. Get the current reward ('r_current'). 
           #      Q_new = CURRENT_REWARD + discount * max(next predicted Q values)
            current_reward = reward[paint_stroke_id]
            Q_new  = current_reward #; Q_newN = current_reward #; Q_newC = current_reward
            # 3.2. (If game is not over), then calculate the Q_new values with the Bellman Eqn.
            if not done[paint_stroke_id]: # equivalent to "if done[paint_stroke_id] == False:"
                # 3.2.1. Input the ('next_state') into the GNN and output the next prediction ('next_Q_mat').
                #        Q_new = current_reward + discount * max(NEXT PREDICTED Q VALUES)
                next_Q_mat = self.model(next_state.x, next_state.edge_index)
                # plt.figure(3, figsize=(3.5, 3)); plt.clf()
                # sns.heatmap(next_Q_mat.detach().numpy(), cmap='gray_r')
                # next_Q_mat = self.model(next_state[paint_stroke_id].x, 
                #                        next_state[paint_stroke_id].edge_index)
                
                # 3.2.2. Define MAX function and get the index of the maximum value of the prediction. 
                #        Q_new = current_reward + discount * MAX(next predicted Q values)             
                method = 1
                if method == 1:
                   Q_max  = self.max_Qval_is_max_entry(prediction= next_Q_mat)
                   #print(' Q max: ', Q_max)
                if method == 2:
                   max_node, max_color = self.max_Qval_is_max_norm(prediction= next_Q_mat)
          
                # 3.2.3. Compute the new Qval(s) given the Bellman Eqn and the MAX value(s).
                #        Q_NEW = current_reward + discount * max(next predicted Q values)
                if method == 1:
                    Q_new  = current_reward + self.gamma * Q_max
                if method == 2:
                    Q_newN = current_reward + self.gamma * max_node
                    Q_newC = current_reward + self.gamma * max_color
            
            # 3.2.4. Update the Qval(s) in the ('target') matrix:
            #   ('target')[row idx, col idx] = new Qval ('Q_new') such that row idx = ('node') idx , col idx = ('color') idx
            method = 1 # check the MAX function method.
            if method == 1:
               target[idxN, idxC -1] = Q_new
            if method == 2:
            # 3.2.4. Update the node Qval ('Q_newN').
            #idxN = torch.argmax(node[paint_stroke_id]).item()
            #target[paint_stroke_id][idxN] = Q_newN
            # 3.2.4. Update the color Qval ('Q_newC').
                idxC = torch.argmax(color_sku).item() -1 
                target[idxN, idxC] = Q_newC
            
        # 4. Compute loss and backpropagate.
        self.optimizer.zero_grad()
        loss = self.criterion(target, Q_mat) # target=Q_new, pred=Q
        print(' loss: ', loss)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()


    """ ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
        Dependent Functions
    !!!!!!!!!!!!!!!!!!!!!!! """

    def max_Qval_is_max_entry(self, prediction: torch.Tensor):
        Q_max = prediction.detach().numpy().max()
        return Q_max
    
    def max_Qval_is_max_norm(self, prediction: torch.Tensor):
        # Method 2.A. Chose the maximum norm amongst the feature vectors
        norm = np.linalg.norm(prediction.detach().numpy(), axis=1)
        #print(' norms of each node feature vectors: ', norm.shape, '\n', norm) 
        max_node  = np.max(norm)
        print(' max node: ', max_node)
        # Method 2.B. Take the max value for the color.
        max_node_idx  = np.argmax(norm)
        max_color = np.max((prediction.detach().numpy())[max_node_idx,:])
        print(' max color: ', max_color)
        return max_node, max_color
    
