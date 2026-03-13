import torch
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
from torch.nn import Linear, Softmax
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from torch_geometric.nn import ChebConv, GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv, CuGraphGATConv,\
                                FusedGATConv, GATv2Conv,TransformerConv, global_mean_pool
 
verbose = False

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        conv_layer_type = GCNConv ##Default: GraphConv / ChebConv
        self.conv1 = conv_layer_type(input_size, hidden_size)#, 2, None)
        self.conv2 = conv_layer_type(hidden_size, hidden_size)#, 2, None)
        self.lin   = Linear(hidden_size, output_size)
        #self.softmax = Softmax(dim=number_of_classes)

    def forward(self, x, edge_index): # took out batch
        print('x1: ', x)
        #print('edge index: ', edge_index)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        print('x2: ', x)
        x = x.relu()
        print('x3: ', x)
        x = self.conv2(x, edge_index)
        print('x4: ', x)

        # 2. Readout layer
        x = global_mean_pool(x)#, batch=torch.tensor(1))  # [batch_size, hidden_size]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.lin(x)
        x = F.softmax(x, dim=1)

        return x

    def save(self, subfolder_name, file_name):
        # Make a folder called 'pretrained_models/' to save runs.
        # 'subfolder_name' details the network convolutional layer types and amount.
        model_folder_path = data_path+'/trained_models/'+ subfolder_name
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f'Saved model {file_name} \n to path {model_folder_path}.')



class QTrainer:
    def __init__(self, model, lr, gamma, game):
        self.model = model
        self.lr    = lr
        self.gamma = gamma # parameter for how much agent cares about rewards in immediate versus distant future.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr) # The way the adam optimizer works is that it takes the parameters of the model and updates them based on the loss function
        self.criterion = torch.nn.MSELoss() # mean squared error: (y - y_pred)^2
        self.num_of_nodes = game.num_of_nodes

    def train_step(self, state, node, color_sku, reward, next_state, done, short_term):
        # Convert the networkX colored graph "state" into a PyTorch graph.
        # The PyTorch graph will be inputted into the GCN and output a vector output.
        state      = from_networkx(G= state,
                                   group_node_attrs= 'color')
        next_state = from_networkx(G= next_state, 
                                   group_node_attrs= 'color')
        node       = torch.tensor(node, dtype=torch.long)
        color_sku  = torch.tensor(color_sku, dtype=torch.long)
        reward     = torch.tensor(reward, dtype=torch.float)

        # (n, x)

        if len(state.shape) == 1: # if the state is a vector, we need to add a dimension to it to make it a matrix of size (1, x) instead of a vector of size (x, )
            # (1, x)
            #state      = torch.unsqueeze(state, 0) # unsqueeze adds a dimension to the tensor
            #next_state = torch.unsqueeze(next_state, 0)
            node       = torch.unsqueeze(node, 0)
            color_sku  = torch.unsqueeze(color_sku, 0)
            reward     = torch.unsqueeze(reward, 0)
            done       = (done, )

        # 1: predicted Q values with current state. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # Q values are the values of each direction for the current state
        pred = self.model(state.x, 
                          state.edge_index) 
        # pass the state to the model to get the predicted Q values
        
        if short_term == True:
            print('pred(state0) - train_step: ', pred)

        target = pred.clone()
        for stroke_idx in range(len(done)): #len(done) equals the number of steps played over all games. ( len(done)= total number of strokes over all games )
            Q_newN = reward[stroke_idx] # Qval for painter's cell.
            Q_newC = reward[stroke_idx] # Qval for color.
            # If game is not over, then calculate the Q_new value.
            if not done[stroke_idx]: # equivalent to "if done[stroke_idx] == False:"
                # Input the new state and obtain its next_pred.
                next_pred = self.model(next_state[stroke_idx].x, 
                                       next_state[stroke_idx].edge_index)
                
                if verbose == True and short_term == True:
                    print(" pred(state1) - train_step: ", next_pred)
                    print(' pred(state1) - direction/next_pred[0:4]: ', next_pred[0:self.num_of_nodes])
                    print(' pred(state1) - color/next_pred[4:]: ', next_pred[self.num_of_nodes: ])
            
            # Define a MAX function.
                max_node   = torch.max(next_pred[0: self.num_of_nodes])
                max_color  = torch.max(next_pred[self.num_of_nodes:]) 
                
                if verbose == True and short_term == True:
                    print(" pred(state1) - max cell Qval: ", max_node.item())
                    print(" pred(state1) - max color Qval: ", max_color.item())
          
           # Compute the new Qval given the Bellman Eqn and the MAX value.
                Q_newN = reward[stroke_idx] + self.gamma * max_node
                Q_newC = reward[stroke_idx] + self.gamma * max_color
                
                if verbose == True and short_term == True:
                    print(" new Qval - cell: ", Q_newN.item())
                    print(" new Qval - color: ", Q_newC.item())      
            if verbose == True and short_term == True:
                print(" pred(state0) - idxP: ", torch.argmax(node[stroke_idx]).item())
            
            # This updates the 'node' Qval.
            idxN = torch.argmax(node[stroke_idx]).item() # Index for the direction.
            target[stroke_idx][idxN] = Q_newN

            if verbose == True and short_term == True:
                print(" pred(state0) - idxC 1: ", torch.argmax(color_sku[stroke_idx]).item())
                print(" pred(state0) - idxC 2: ", self.num_of_nodes + torch.argmax(color_sku[stroke_idx]).item())
            
            # This updates the 'color choice' Qval.
            idxC = torch.argmax(color_sku[stroke_idx]).item()
            target[stroke_idx][self.num_of_nodes + idxC] = Q_newC # The color index starts on 'self.num_of_nodes'.
            
            if verbose == True and short_term == True:
                print('target: ', target)
                print('\n')

    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(direction)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # target=Q_new, pred=Q
        loss.backward()

        self.optimizer.step()


