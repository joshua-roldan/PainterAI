import torch
from torch_geometric.data import Data
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import helper
import time

verbose = False


class GCNConv(MessagePassing):
    def __init__(self, input_size, output_size):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin  = Linear(input_size, output_size, bias=False)
        self.bias = Parameter(torch.empty(output_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index 
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    



class QTrainer:
    def __init__(self, model, lr, gamma, game):
        self.model = model
        self.lr    = lr
        self.gamma = gamma # parameter for how much agent cares about rewards in immediate versus distant future.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr) # The way the adam optimizer works is that it takes the parameters of the model and updates them based on the loss function
        self.criterion = torch.nn.MSELoss() # mean squared error: (y - y_pred)^2
        self.num_of_nodes = game.num_of_nodes

    # 'train_step()' :: This method takes the colored graphs and trains the network to predict the right Qvals.
    def train_step(self, state, next_state, node_sku, color_sku, reward, done, short_term):
        # 0. Suppose a graph time series...
        # 1. Convert "node_sku", "color_sku"
        #    b/c 
        # 2. Output model's prediction given (PyTorch) "state",
        #    i.e. predicted Q values with current state. 
        #    Q values are the values of each color for the current state.
        #    Note that ('pred'), and thus ('target'), is a nxm matrix
        #    where n= # of nodes, m= # of colors.
        #    2.1. Input ('state') and output the prediction matrix ('pred').
        #    2.2. Create a copy ('target') from ('pred').
        # 3. Update Qvals:
        #    Q_new = r + y * max(next_predicted Q value) -> only do this if not done    
        #    3.1. Get the preliminary Qvals for the node ('Q_newN') and color ('Q_newC'). 
        #    3.2. (If game is not over), then calculate the Q_new values.    
        #         3.2.1. Input the ('next_state') into the GCN, and output the next prediction ('next_pred').
        #         3.2.2. Define MAX function and get the max values.
        #                There are two methods for max node ('max_node') and max color ('max_color'):
        #                 - Method 1: Take the max Qval of the entire matrix.
        #                 - Method 2: A. Take the max norm among node feature vectors,
        #                             B. Take the max value for the color.
        #         3.2.3. Compute the new Qval(s) given the Bellman Eqn and the MAX value(s).
        #         3.2.4. Update the Qval(s) in the ('target') matrix:
        #                ('target')[row idx, col idx] = new Qval ('Q_new').
        #                s.t. row idx = ('node') idx , col idx = ('color') idx
        #    (Repeat for ?)
        # 4. Compute loss and backpropagate.
        #      
        print( 'training...')
    
        # 1. Convert ('node_sku'), ('color_sku'), ('reward').
        print('node_sku (check): ', type(node_sku))
        idxN       = helper.convert_sku_to_int(node_sku)
        print('node idx: ', idxN)
        node_sku   = torch.tensor(node_sku, dtype=torch.long)
        color_sku  = torch.tensor(color_sku, dtype=torch.long)
        reward     = torch.tensor(reward, dtype=torch.float)

        node_sku   = torch.unsqueeze(node_sku, 0)
        color_sku  = torch.unsqueeze(color_sku, 0)
        reward     = torch.unsqueeze(reward, 0)
        done       = (done, )

        # 2. Output model's prediction given "state",
        #    i.e. predicted Q values with current state. 
        #    Q values are the values of each color for the current state.
        #    Note that ('pred'), and thus ('target'), is a nxm matrix
        #    where n= # of nodes, m= # of colors.
        #  2.1 Input ('state') and output the prediction matrix ('pred').
        pred   = self.model(state.x, 
                            state.edge_index)
        #  2.2 Create a copy ('target') from ('pred').
        target = pred.clone()

        if short_term == True:
            print('\n')
            print('pred(state) - train_step: ', pred.shape, '\n', pred)

        # 3. Update Qvals.
        #    Q_new = r + y * max(next_predicted Q value) -> only do this if not done    
        for paint_stroke_id in range(len(done)): #len(done) equals the number of steps played over all games. ( len(done)= total number of strokes over all games )
            # 3.1. Get the preliminary Qvals for the node ('Q_newN') and color ('Q_newC').
            Q_newN = reward[paint_stroke_id]
            Q_newC = reward[paint_stroke_id]
            # 3.2. (If game is not over), then calculate the Q_new values.
            if not done[paint_stroke_id]: # equivalent to "if done[paint_stroke_id] == False:"
                # 3.2.1. Input the ('next_state') into the GCN and output the next prediction ('next_pred').
                next_pred = self.model(next_state.x,
                                       next_state.edge_index)
                #print('next pred (check): ', next_pred.shape, '\n', next_pred)
                # next_pred = self.model(next_state[paint_stroke_id].x, 
                #                        next_state[paint_stroke_id].edge_index)
                
                if verbose == True and short_term == True:
                    print(" pred(next_state) - next_pred: ", next_pred)
                    print(' pred(next_state) - direction/next_pred[0:4]: ', next_pred[0:self.num_of_nodes])
                    print(' pred(next_state) - color/next_pred[4:]: ', next_pred[self.num_of_nodes: ])
            
                # 3.2.2. Define MAX function and get the index of the maximum value of the prediction.
                # We use Method 2 below:
                # Method 2.A. Chose the maximum norm amongst the feature vectors
                norm = np.linalg.norm(next_pred.detach().numpy(), axis=1)
                #print(' norms of each node feature vectors: ', norm.shape, '\n', norm) 
                max_node  = np.max(norm)
                print(' max node: ', max_node)
                # Method 2.B. Take the max value for the color.
                max_node_idx  = np.argmax(norm)
                max_color = np.max((next_pred.detach().numpy())[max_node_idx,:])
                print(' max color: ', max_color)

                if verbose == True and short_term == True:
                    print(" pred(state1) - max node Qval: ", max_node.item())
                    print(" pred(state1) - max color Qval: ", max_color.item())
          
                # 3.2.3. Compute the new Qval(s) given the Bellman Eqn and the MAX value(s).
                Q_newN = reward[paint_stroke_id] + self.gamma * max_node
                Q_newC = reward[paint_stroke_id] + self.gamma * max_color
                
                if verbose == True and short_term == True:
                    print(" new Qval - node: ", Q_newN.item())
                    print(" new Qval - color: ", Q_newC.item())      
            if verbose == True and short_term == True:
                print(" pred(state0) - idxP: ", torch.argmax(node_sku[paint_stroke_id]).item())
            
            # 3.2.4. Update the node Qval ('Q_newN').
            #idxN = torch.argmax(node[paint_stroke_id]).item() # Index for the direction.
            #target[paint_stroke_id][idxN] = Q_newN

            # 3.2.4. Update the color Qval ('Q_newC').
            idxC = torch.argmax(color_sku).item() - 1
            target[idxN, idxC] = Q_newC
            
        # 4. Compute loss and backpropagate.
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # target=Q_new, pred=Q
        print('loss: ', loss)
        loss.backward()

        self.optimizer.step()


