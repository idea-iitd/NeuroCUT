import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.utils import k_hop_subgraph
from utils import *
import pickle as pkl
import matplotlib.pyplot as plt
import time
 
 
 
class ModelBasic(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, model_name , num_features,device):
        super().__init__()
        torch.manual_seed(1234567)
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)
 
        func = None
        if model_name == 'gcn':
            func = GCNConv
        elif model_name == 'gat':
            func = GATConv
        elif model_name == 'graphsage':
            func = SAGEConv
        elif model_name == 'gin':
            func = GINConv
        else:
            print('model name not found')
            exit(0)
 
        self.conv_layers = []
        self.batch_norms = []
        self.conv_layers.append(func(num_features, hidden_channels).to(device).double())
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        for i in range(num_layers-1):
            self.conv_layers.append(func(hidden_channels, hidden_channels).to(device).double())
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        
        # mlp
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
 
    
 
    def forward(self, x, edge_index, curr_node_id ,partitions, core_values):
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue
        # # print(x.shape)
        # print(edge_index.shape)
        # print(x, edge_index)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = x.relu()            
            # x = F.dropout(x, p=0.5, training=self.training)
        
        # x -> node embeddings
 
        num_partitions = len(partitions)
 
        partition_embedding = self.getPartitonEmbedding(x, partitions) # partition embeddings:  (num_partitions, hiddent_channels)
        
 
        x_curr = x[curr_node_id].reshape((1,x.shape[1])) # current node embedding. shape = (1, hidden_channels)
 
 
        # concat current node embedding and partition embeddings. shape = (num_partitions, 2*hidden_channels)
        x = torch.cat([x_curr.expand(num_partitions,-1), partition_embedding], dim=1) # concat current node embedding and partition embedding
 
        #mlp
        x = self.lin1(x)
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin2(x)
 
        # sigoid?
        # x = x.sigmoid()
 
        return x
    
    def getPartitonEmbedding(self, x, partitions):        # sum pool or mean pool or max pool or some mlp?
 
        # partitions is a dictinary with key as partition id and value as list of nodes in that partition
        # x is the node embedding matrix
        # return a matrix of size (num_partitions, hidden_channels)
 
        device = x.device
 
        num_partitions = len(partitions)
        
        # sum pooling
        partition_embedding = torch.zeros(num_partitions, x.shape[1]).to(device)
        for i in range(num_partitions):
            partition_embedding[i] = torch.sum(x[partitions[i]], dim=0)
        
        return partition_embedding
    
 
class ModelAtten(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, model_name , num_features,device):
        super().__init__()
        torch.manual_seed(1234567)
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)
 
        func = None
        if model_name == 'gcn':
            func = GCNConv
        elif model_name == 'gat':
            func = GATConv
        elif model_name == 'graphsage':
            func = SAGEConv
        elif model_name == 'gin':
            func = GINConv
        else:
            print('model name not found')
            exit(0)
 
        self.conv_layers = []
        self.batch_norms = []
        self.conv_layers.append(func(num_features, hidden_channels).to(device).double())
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        for i in range(num_layers-1):
            self.conv_layers.append(func(hidden_channels, hidden_channels).to(device).double())
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        
        
        # mlp
        self.lin1 = Linear(3*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
 
    
 
    def forward(self, x, edge_index, curr_node_id ,partitions, core_values):
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue
        # # print(x.shape)
        # print(edge_index.shape)
        # print(x, edge_index)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = x.relu()            
            # x = F.dropout(x, p=0.5, training=self.training)
        
        # x -> node embeddings
 
        num_partitions = len(partitions)
 
        partition_embedding = self.getPartitonEmbedding_usingCoreValue(x, partitions, core_values) # partition embeddings:  (num_partitions, hiddent_channels)
        
 
        x_curr = x[curr_node_id].reshape((1,x.shape[1])) # current node embedding. shape = (1, hidden_channels)
 
 
        # concat current node embedding and partition embeddings. shape = (num_partitions, 2*hidden_channels)
        x = torch.cat([x_curr.expand(num_partitions,-1), partition_embedding], dim=1) # concat current node embedding and partition embedding
 
 
 
        #mlp
        x = self.lin1(x)
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin2(x)
 
        # sigoid?
        # x = x.sigmoid()
 
        return x
    
    
    def getPartitonEmbedding_usingCoreValue(self, x, partitions, core_values):        # sum pool or mean pool or max pool or some mlp?
 
        # partitions is a dictinary with key as partition id and value as list of nodes in that partition
        # x is the node embedding matrix
        # return a matrix of size (num_partitions, hidden_channels)
 
        device = x.device
 
        num_partitions = len(partitions)
        
        # core weighted sum pooling
        partition_embedding_core = torch.zeros(num_partitions, x.shape[1]).to(device)
        for i in range(num_partitions):
            # TODO also check if core_values is 0 indexed or just like dictionary?
            partition_embedding_core[i] = torch.sum(x[partitions[i]]*core_values[partitions[i]].reshape(-1,1), dim=0) # TODO check it
               
 
        # non-core weighted sum pooling
        partition_embedding_non_core = torch.zeros(num_partitions, x.shape[1]).to(device)
        for i in range(num_partitions):
            partition_embedding_non_core[i] = torch.sum(x[partitions[i]]*(1-core_values[partitions[i]]).reshape(-1,1), dim=0)
        
        # concat core and non-core
        partition_embedding = torch.cat([partition_embedding_core, partition_embedding_non_core], dim=1)
        
        return partition_embedding
    
 
class ModelLinkPred(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, model_name, num_features,scoring_func='mlp', device='cpu'):
        super().__init__()
        torch.manual_seed(1234567)
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)
 
        func = None
        if model_name == 'gcn':
            func = GCNConv
        elif model_name == 'gat':
            func = GATConv
        elif model_name == 'graphsage':
            func = SAGEConv
        elif model_name == 'gin':
            func = GINConv
        else:
            print('model name not found')
            exit(0)
 
        self.conv_layers = []
        self.batch_norms = []
        self.conv_layers.append(func(num_features, hidden_channels).to(device).double())
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        for i in range(num_layers-1):
            self.conv_layers.append(func(hidden_channels, hidden_channels).to(device).double())
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        
        # mlp
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
 
        # scoring function
        self.scoring_func = scoring_func
        print('scoring function: ', scoring_func)
 
    
 
    def forward(self, x, edge_index, curr_node_id ,partitions,pool,node_weights, embeddings=None):      # node_weights not used here
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue
        # # print(x.shape)
        # print(edge_index.shape)
        # print(x, edge_index)
    
        if embeddings is not None:
            x = embeddings
        else:
            for i in range(len(self.conv_layers)):
                x = self.conv_layers[i](x, edge_index)
                x = self.batch_norms[i](x)
                x = x.relu()
            # x = F.dropout(x, p=0.5, training=self.training)
        
        # x -> node embeddings
 
        x_curr = x[curr_node_id].reshape((1,x.shape[1])) # current node embedding. shape = (1, hidden_channels)
        num_partitions = partitions.shape[1]
        # print(x.shape, x_curr.shape)
        #         # get score for every node. score = self.mlp(x,x_curr)
        scores = self.getScore(x, x_curr) # shape = (num_nodes, 1)
 
        # get partition scores of shape (1, num_partitions)
        if pool=='sum':
            partition_scores = torch.matmul(scores.t(), partitions) # shape = (1, num_partitions)
        elif pool=='mean':
            partition_scores = torch.matmul(scores.t(), partitions)/(torch.sum(partitions, dim=0).reshape((1,num_partitions))+1) # shape = (1, num_partitions)
        elif pool=='max':
            partition_scores, _ = torch.max(scores*partitions, dim=0) # shape = (1, num_partitions)
        else:
            print('pooling function not found')
            exit(0)
        
        
 
 
        # partitions is a matrix of shape (num_nodes, num_partitions)
        # partition_scores = torch.matmul(partitions.t(), scores)/(torch.sum(partitions, dim=0).reshape((num_partitions,1))+1) # shape = (num_partitions, 1)
        # # partition_scores should be divided by total number of nodes in partition
        if embeddings is None:
            return partition_scores, x
        else:
            return partition_scores, None  # dont care
 
    
    def getScore(self, x, x_curr):
        # x is node embedding matrix
        # x_curr is current node embedding
 
        if self.scoring_func == 'mlp':
            return self.mlp(x, x_curr)
 
        elif self.scoring_func == 'dot':
            score = torch.matmul(x, x_curr.t())
            return score
        
        elif self.scoring_func == 'cosine':
            score = F.cosine_similarity(x, x_curr.repeat(x.shape[0],1), dim=1)
            return score
        
        elif self.scoring_func == 'l2':
            score = torch.norm(x-x_curr.repeat(x.shape[0],1), p=2,dim=1)
            return score
        
        elif self.scoring_func == 'l1':
            score = torch.norm(x-x_curr.repeat(x.shape[0],1), p=1, dim=1)
            return score
        
        else:
            print('scoring function not found')
            exit(0)
 
    
    def mlp(self, x, x_curr):
        # x is node embedding matrix
        # x_curr is current node embedding
 
        # concat x and x_curr
        x = torch.cat([x, x_curr.repeat(x.shape[0],1)], dim=1)
 
        # mlp
        x = self.lin1(x)
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin2(x)
 
        return x
    
class ModelLocalLinkPred(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, model_name, num_features,hops=2,scoring_func='mlp', device='cpu'):
        super().__init__()
        torch.manual_seed(1234567)
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)
 
        func = None
        if model_name == 'gcn':
            func = GCNConv
        elif model_name == 'gat':
            func = GATConv
        elif model_name == 'graphsage':
            func = SAGEConv
        elif model_name == 'gin':
            func = GINConv
        else:
            print('model name not found')
            exit(0)
 
        self.hops=hops
 
        self.conv_layers = []
        self.batch_norms = []
        self.conv_layers.append(func(num_features, hidden_channels).to(device).double())
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        for i in range(num_layers-1):
            self.conv_layers.append(func(hidden_channels, hidden_channels).to(device).double())
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        
        # mlp
        self.lin1 = Linear(2*hidden_channels+1, hidden_channels)        # +1 due to node weight product
        self.lin2 = Linear(hidden_channels, 1)
 
        # scoring function
        self.scoring_func = scoring_func
        print('scoring function: ', scoring_func)
 
    
 
    def forward(self, x, edge_index, curr_node_id ,partitions,pool, node_weights ,embeddings=None):
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue
        # # print(x.shape)
        # print(edge_index.shape)
        # print(x, edge_index)
        if embeddings is not None:
            x = embeddings
        else:
            for i in range(len(self.conv_layers)):
                x = self.conv_layers[i](x, edge_index)
                x = self.batch_norms[i](x)
                x = x.relu()            
            # x = F.dropout(x, p=0.5, training=self.training)
        
        # x -> node embeddings
        x_curr = x[curr_node_id].reshape((1,x.shape[1])) # current node embedding. shape = (1, hidden_channels)
        num_partitions = partitions.shape[1]
 
        # get scores by using only my neighbors
        # neighbors = edge_index[0][edge_index[1]==curr_node_id]
        # khop neighbors
        k = self.hops
        neighbors,_,_,_ = k_hop_subgraph(curr_node_id.item(), k, edge_index, num_nodes=x.shape[0])
        
        # remove curr_node_id
        neighbors = neighbors[neighbors!=curr_node_id.item()]
        
 
        x_neighbors = x[neighbors]
        # print(x_neighbors.shape, x_curr.shape)
        scores = self.getScore(x_neighbors, x_curr, node_weights[neighbors], node_weights[curr_node_id]) # shape = (num_neighbors, 1)
        
        
 
 
        # get partition scores of shape (1, num_partitions)
        if pool=='sum':
            partition_scores = torch.matmul(scores.t(), partitions[neighbors]) # shape = (1, num_partitions)
        elif pool=='mean':
            partition_scores = torch.matmul(scores.t(), partitions[neighbors])/(torch.sum(partitions[neighbors], dim=0).reshape((1,num_partitions))+1) # shape = (1, num_partitions)
        elif pool=='max':
            partition_scores, _ = torch.max(scores*partitions[neighbors], dim=0) # shape = (1, num_partitions)
        else:
            print('pooling function not found')
            exit(0)
 
        
 
        if embeddings is None:
            return partition_scores, x
        else:
            return partition_scores, None  # dont care
 
    
    def getScore(self, x, x_curr, node_weights, node_weights_curr):
        # x is node embedding matrix
        # x_curr is current node embedding
 
        if self.scoring_func == 'mlp':
            return self.mlp(x, x_curr, node_weights, node_weights_curr)
 
        elif self.scoring_func == 'dot':
            score = torch.matmul(x, x_curr.t())
            return score
        
        elif self.scoring_func == 'cosine':
            score = F.cosine_similarity(x, x_curr.repeat(x.shape[0],1), dim=1)
            return score
        
        elif self.scoring_func == 'l2':
            score = torch.norm(x-x_curr.repeat(x.shape[0],1), p=2,dim=1)
            return score
        
        elif self.scoring_func == 'l1':
            score = torch.norm(x-x_curr.repeat(x.shape[0],1), p=1, dim=1)
            return score
        
        else:
            print('scoring function not found')
            exit(0)
 
    
    def mlp(self, x, x_curr, node_weights, node_weights_curr):
        # x is node embedding matrix
        # x_curr is current node embedding
 
        # concat x and x_curr
        x = torch.cat([x, x_curr.repeat(x.shape[0],1)], dim=1)
 
        # Expand dimensions of node_weight tensor for broadcasting
        expanded_node_weights = node_weights.unsqueeze(1)  # Shape: num_neighboursx1
 
        # Multiply all values in expanded_node_weight by weight_curr
        expanded_node_weights *= node_weights_curr
 
        x = torch.cat([x,expanded_node_weights], dim=1)
 
        # mlp
        x = self.lin1(x)
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin2(x)
 
        return x



class ModelLinkPred_weight(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, model_name, num_features,scoring_func='mlp', device='cpu'):
        super().__init__()
        torch.manual_seed(1234567)
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)
        self.device = device
        func = None
        # if model_name == 'gcn':
        #     func = GCNConv
        # elif model_name == 'gat':
        #     func = GATConv
        # elif model_name == 'graphsage':
        #     func = SAGEConv
        # elif model_name == 'gin':
        #     func = GINConv
        # else:
        #     print('model name not found')
        #     exit(0)

        func = GCNConv
        
 
        self.conv_layers = []
        self.batch_norms = []
        self.conv_layers.append(func(num_features, hidden_channels).to(device).double())
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        for i in range(num_layers-1):
            self.conv_layers.append(func(hidden_channels, hidden_channels).to(device).double())
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device))
        
        # mlp
        self.lin1 = Linear(4*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
 
        # scoring function
        self.scoring_func = scoring_func
        print('scoring function: ', scoring_func)
 
    
 
    def forward(self, x, edge_index, curr_node_id ,partitions,pool,node_weights, embeddings=None):      # node_weights not used here
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue
        # # print(x.shape)
        # print(edge_index.shape)
        # print(x, edge_index)
        # TODO for horse
        # get edge_weights here only
        import scipy.sparse
        data = scipy.sparse.load_npz('/DATATWO/users/mincut/BTP-Final/data_new/horse/horse_sparse.npz')
        data_T = torch.tensor(data.todense()).to(self.device)

        data = data.tocoo()
        # a = np.vstack([data.row, data.col])
        edge_weights = torch.tensor(data.data,dtype=torch.float).to(self.device)

        if embeddings is not None:
            x = embeddings
        else:
            for i in range(len(self.conv_layers)):
                x = self.conv_layers[i](x, edge_index, edge_weight=edge_weights)
                x = self.batch_norms[i](x)
                x = x.relu()
            # x = F.dropout(x, p=0.5, training=self.training)
        
        # x -> node embeddings
 
        x_curr = x[curr_node_id].reshape((1,x.shape[1])) # current node embedding. shape = (1, hidden_channels)
        num_partitions = partitions.shape[1]
        # print(x.shape, x_curr.shape)
        #         # get score for every node. score = self.mlp(x,x_curr)
        # get x_curr_weights = data_T[curr_node_id]
        scores = self.getScore(x, x_curr,data_T[curr_node_id]) # shape = (num_nodes, 1)      # TODO maybe use edge weight here also

        # get partition scores of shape (1, num_partitions)
        if pool=='sum':
            partition_scores = torch.matmul(scores.t(), partitions) # shape = (1, num_partitions)
        elif pool=='mean':
            partition_scores = torch.matmul(scores.t(), partitions)/(torch.sum(partitions, dim=0).reshape((1,num_partitions))+1) # shape = (1, num_partitions)
        elif pool=='max':
            partition_scores, _ = torch.max(scores*partitions, dim=0) # shape = (1, num_partitions)
        else:
            print('pooling function not found')
            exit(0)
        
        
 
 
        # partitions is a matrix of shape (num_nodes, num_partitions)
        # partition_scores = torch.matmul(partitions.t(), scores)/(torch.sum(partitions, dim=0).reshape((num_partitions,1))+1) # shape = (num_partitions, 1)
        # # partition_scores should be divided by total number of nodes in partition
        if embeddings is None:
            return partition_scores, x
        else:
            return partition_scores, None  # dont care
 
    
    def getScore(self, x, x_curr, edge_curr_node):
        # x is node embedding matrix
        # x_curr is current node embedding
 
        if self.scoring_func == 'mlp':
            return self.mlp(x, x_curr,edge_curr_node)
 
        elif self.scoring_func == 'dot':
            score = torch.matmul(x, x_curr.t())
            return score
        
        elif self.scoring_func == 'cosine':
            score = F.cosine_similarity(x, x_curr.repeat(x.shape[0],1), dim=1)
            return score
        
        elif self.scoring_func == 'l2':
            score = torch.norm(x-x_curr.repeat(x.shape[0],1), p=2,dim=1)
            return score
        
        elif self.scoring_func == 'l1':
            score = torch.norm(x-x_curr.repeat(x.shape[0],1), p=1, dim=1)
            return score
        
        else:
            print('scoring function not found')
            exit(0)
 
    
    def mlp(self, x, x_curr,edge_curr_node):
        # x is node embedding matrix
        # x_curr is current node embedding
        # concat x and x_curr
        x = torch.cat([x, x_curr.repeat(x.shape[0],1)], dim=1)

        x = torch.cat([x,edge_curr_node.unsqueeze(1).repeat(1,x.shape[1])],dim=1)
 
        # mlp
        x = self.lin1(x)
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin2(x)
 
        return x
    
 
 
 
def testtt():
    
    pass