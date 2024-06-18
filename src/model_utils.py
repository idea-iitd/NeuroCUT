import torch
from tqdm import tqdm
import numpy as np
import math
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,OPTICS
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import random
from torch_geometric.utils import from_networkx,to_networkx
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric,type_metric
from pyclustering.cluster.center_initializer import random_center_initializer,kmeans_plusplus_initializer
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import sys

def getPartitions(model,data,num_cuts,isAssigned_feature='false'):
    model.eval() 
    
    with torch.no_grad():
        num_nodes = data.num_nodes
        node_feat = data.x[0].clone()           # TODO check this
        partitions = {}
        num_partitions = num_cuts          # TODO how to make this generic
        for i in range(num_partitions):
            partitions[i] = []

        for i in tqdm(range(num_nodes)):
            curr_node_id = int(data.x[1][i].item())            # data.x[1] is the node_ordering
            out = model(node_feat, data.edge_index, curr_node_id, partitions, data.x[2]).reshape((1,num_cuts)) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
            # update node feature of current node as assigned to a partition
            if isAssigned_feature.lower()=='true':
                # update node feature of current node as assigned to a partition
                node_feat[curr_node_id][node_feat.shape[1]-1] = 1.0         # TODO check this
                
            predicted_partition = torch.argmax(out)
            partitions[predicted_partition.item()].append(curr_node_id)          # update the partition

    return partitions


def getCutValue(data,partitions,device):
    # partitions -> n*k
    edge_idx = data.edge_index.cpu()
    # get adjacency matrix from edge_idx
    A=nx.adjacency_matrix(to_networkx(data))
    # A = nx.adjacency_matrix(nx.from_edgelist(edge_idx.t().numpy()))
    A = torch.from_numpy(A.todense()).double().to(device)

    cut_sum = torch.sum(torch.mm(partitions,(1-partitions).t())*A).to(device)
    tot_edges = torch.sum(A).to(device)
    return cut_sum/tot_edges

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def getNormalisedCutValue(data,partitions,num_cuts,device,cuttype,default=float('NaN')):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    Y = partitions
    # get sparse adjecency matrix from edge_index
    A = torch.sparse_coo_tensor(data.edge_index.to(device), torch.ones(data.edge_index.shape[1]).to(device), (data.num_nodes, data.num_nodes)).to(device)
    # A=sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(to_networkx(data))).to(device)           # TODO make this again?
    if cuttype=='normalised':
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).double())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        normalised_cut = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
        ## TODO : check this 
        # print(default)
    elif cuttype=='normalised_sparse':


        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).double())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()

        YbyGamma = YbyGamma.to_sparse()
        Y_t = Y_t.to_sparse()
        normalised_cut = torch.sparse.sum(torch.sparse.mm(YbyGamma, Y_t) * A).to(device)
    
    elif cuttype=='normalised_weight':


        import scipy.sparse

        A = scipy.sparse.load_npz('/DATATWO/users/mincut/BTP-Final/data_new/horse/horse_sparse.npz')



        # # Convert the adjacency matrix to COO format
        A = A.tocoo()


        # # Compute the degree matrix D
        # degree = torch.tensor(A.sum(axis=1)).sqrt()  # Compute the square root of the sum of each row
        # D_inv_sqrt = torch.diag(torch.squeeze(1.0 / degree))  # Compute the inverse square root of the degree matrix


        # A = torch.tensor(A.todense())

        # # Compute the normalized adjacency matrix
        # hat_A = D_inv_sqrt @ A @ D_inv_sqrt

        # # Print the normalized adjacency matrix
        # print(hat_A)

        # A = hat_A.to_sparse().to(device)

        A = torch.tensor(A.todense()).to_sparse().to(device)


        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).double())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()

        YbyGamma = YbyGamma.to_sparse()
        Y_t = Y_t.to_sparse()
        # a1 = torch.mm(YbyGamma.to('cpu'), Y_t.to('cpu'))
        # a2 = A.to('cpu').to_dense()
        # normalised_cut = torch.sum(a1 * a2).to(device)
        normalised_cut = torch.sparse.sum(torch.sparse.mm(YbyGamma, Y_t) * A).to(device)
        ## TODO : check this 
        # print(default)
    elif cuttype=='kmin':
        Y_t = (1 - Y).t()
        k_cut = torch.sum(torch.mm(Y, Y_t) * A.to_dense()).to(device)
        num_edge=data.num_edges
        normalised_cut=k_cut/num_edge
    
    elif cuttype=='sparsest':
        n=A.shape[0]
        count=torch.sum(Y,axis=0)
        gamma=torch.min(n-count,count) ## min |S|,|SBAR|
        Ybygamma=torch.div(Y,gamma) 
        Y_t = (1 - Y).t()
        k_cut = torch.sum(torch.mm(Ybygamma, Y_t) * A.to_dense()).to(device)
        normalised_cut=k_cut
    elif cuttype=='sparsest_weight':
        weights=(data.node_weights*Y.t()).t()
        weight_count=torch.sum(weights,axis=0)
        total_weight=torch.sum(weight_count)
        gamma=torch.min(total_weight-weight_count,weight_count) ## min |S|,|SBAR|
        Ybygamma=torch.div(Y,gamma) 
        Y_t = (1 - Y).t()
        k_cut = torch.sum(torch.mm(Ybygamma, Y_t) * A.to_dense()).to(device)
        normalised_cut=k_cut
    elif cuttype=='GAP':
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).double())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        l1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
        oneT = torch.ones((1, Y.shape[0])).double().to(device)
    # # Loss2 is normalizing term
        n = Y.shape[0]
        if isinstance(num_cuts, torch.Tensor):
            g = num_cuts.item()
        else:
            g = num_cuts
        # l2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t()))
        l2 = torch.sum((torch.mm(oneT,Y) - (n/g))**2)
        normalised_cut = l1+l2
    elif cuttype=='GAP_1':
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).double())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        l1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
        oneT = torch.ones((1, Y.shape[0])).double().to(device)
    # # Loss2 is normalizing term
        n = Y.shape[0]
        if isinstance(num_cuts, torch.Tensor):
            g = num_cuts.item()
        else:
            g = num_cuts
        # l2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t()))
        l2 = torch.sum((torch.mm(oneT,Y) - (n/g))**2)
        l2 = l2/(n**2)
        normalised_cut = l1+l2
    elif cuttype=='GAP_2':
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).double())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        l1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
        oneT = torch.ones((1, Y.shape[0])).double().to(device)
    # # Loss2 is normalizing term
        n = Y.shape[0]
        node_per_partition = torch.mm(oneT,Y)
        node_per_partition = node_per_partition/n
        product = torch.prod(node_per_partition)
        # l2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t()))
        l2 = product
        normalised_cut = l1-l2
    elif cuttype=='conductance':
        D = torch.sparse.sum(A, dim=1).to_dense()
        tot_edge = torch.sum(D)
        gamma = torch.mm(Y.t(), D.unsqueeze(1).double())
        Ybygamma=torch.div(Y,gamma.t()) 
        Y_t = (1 - Y).t()
        phi_node = torch.sum(torch.mm(Ybygamma, Y_t) * A.to_dense(),axis=0)
        phi = torch.mm(Y.t(),phi_node.double().unsqueeze(1))
        ind = torch.nonzero(gamma.squeeze() < tot_edge.item()/2).squeeze()
        phi_ind = phi[ind]
        normalised_cut = torch.min(phi_ind)

    elif cuttype=='mincutpool':
        # TODO only for horse
        import scipy.sparse
        A = scipy.sparse.load_npz('/DATATWO/users/mincut/BTP-Final/data_new/horse/horse_sparse.npz')
        A = torch.tensor(A.todense()).to(device)

        def _rank3_trace(x):
            return torch.einsum('ijj->i', x)


        def _rank3_diag(x):
            eye = torch.eye(x.size(1)).type_as(x)
            out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

            return out

        adj = A.to_dense().double()
        s = Y
        k=s.size(-1)

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s


        s = torch.softmax(s, dim=-1)

        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # MinCut regularization.
        mincut_num = _rank3_trace(out_adj)
        d_flat = torch.einsum('ijk->ij', adj)
        d = _rank3_diag(d_flat)
        mincut_den = _rank3_trace(
            torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
        mincut_loss = -(mincut_num / mincut_den)
        mincut_loss = torch.mean(mincut_loss)

        # Orthogonality regularization.
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        return ortho_loss+mincut_loss

    elif cuttype=='modularity':
        # TODO calculate time
        from time import time
        num_nodes = A.shape[0]
        num_edges = int(len(A._indices()[0])/ 2)  # Assuming undirected graph

        partition_ids = torch.argmax(Y,dim=1)

        D = torch.sparse.sum(A, dim=1).unsqueeze(1)
        modularity = 0.0


        mask_same_community = (partition_ids.view(-1, 1) == partition_ids.view(1, -1)).int()    

        x = (A - (D @ D.t()) / (2 * num_edges)).to_dense()


        modularity = (x * mask_same_community)
        # from tqdm import tqdm

        # for i in tqdm(range(num_nodes)):
        #     for j in range(num_nodes):
        #         if partition_ids[i] == partition_ids[j]:
        #             A_ij = A[i, j]
        #             k_i = D[i]
        #             k_j = D[j]

        #             modularity += A_ij - (k_i * k_j) / (2 * num_edges)

        modularity = torch.sum(modularity) / (2 * num_edges)

        # print("Times Modularity :",t1-t,t2-t1,t3-t2)

        # modularity /= (2 * num_edges)


        normalised_cut = modularity.to(torch.float64)

    else:
        print("Wrong type")
        exit(0)
    if math.isnan(normalised_cut):
        return torch.tensor(default)
    return normalised_cut
    
def gap_loss_sparse(Y, edge_index,device):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    # get sparse adjecency matrix from edge_index
    A = torch.sparse_coo_tensor(edge_index.to(device), torch.ones(edge_index.shape[1]).to(device), (Y.shape[0], Y.shape[0])).to(device)
    n = Y.shape[0]
    g = Y.shape[1]
    # print(torch.min(Y),torch.max(Y))
    D = torch.sparse.sum(A, dim=1).to_dense().to(device)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.tensor([0.], requires_grad=True).to(device)
    # for i in range(idx.shape[1]):
    #     loss = loss.clone() +  torch.dot(YbyGamma[idx[0,i],:], Y_t[:,idx[1,i]])*data[i]
    ## LOSS 1 is cutloss term
    loss1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
    # oneT = torch.ones((1, Y.shape[0])).to(device)
    # # Loss2 is normalizing term
    # # TODO remove loss 2(not in ICLR paper)
    # loss2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t())).to(device)
    # loss2 = torch.sum((torch.mm(oneT,Y) - n/g)**2).to(device)
    # print("Loss1: {}, Loss2: {}".format(100*loss1,0.01*loss2))
    loss=loss1
    return loss


def intialise_assgiment(data,intial_type,device):
    ## TODO: add k-means initialisation
    ## currently random initialisation
    ## returns partitions and normalised_cut
    num_nodes = data.num_nodes
    num_cuts = data.num_cuts
    # make partitions tensor double
    partitions = torch.zeros((num_nodes,num_cuts)).to(device).double()
    if intial_type=='random':
        for i in range(num_nodes):
            part=int(np.random.randint(0,num_cuts))
            partitions[i][part]=1
    elif intial_type=='kmeans':
        features=data.x[0].cpu()
        # print(features.shape)
        # print(type(features))
        kmean = KMeans(n_clusters=num_cuts, random_state=0).fit(features)
        # kmeans = KMeans(n_clusters=num_cuts).fit(features)
        for i in range(num_nodes):
            part=kmean.labels_[i]
            partitions[i][part]=1

    elif intial_type=='kmeansSpectral':
        features=torch.load('../data/cora_lc/train_set/1/features_spectral_5_None.pkl')
        kmean = KMeans(n_clusters=num_cuts, random_state=0).fit(features)
        for i in range(num_nodes):
            part=kmean.labels_[i]
            partitions[i][part]=1
    elif intial_type=='kmeans_Linf':
        custom_metric=distance_metric(type_metric.CHEBYSHEV)
        features=data.x[0].cpu()
        # breakpoint()
        initial_centers=kmeans_plusplus_initializer(features,num_cuts,random_state=0).initialize()
        kmeans_instance=kmeans(features,initial_centers,metric=custom_metric)
        kmeans_instance.process()
        clusters=kmeans_instance.get_clusters()
        # print(clusters)  n ## list of list

        ## tranforming clusters to torch tensor
        # cuts=torch.zeros(features.shape[0])
        for i in range(len(clusters)):
            for j in clusters[i]:
                partitions[j][i]=1
            # partitions[clusters[i]][i]=1
        # print(partitions)
    elif intial_type=='dbscan':
        # TODO select the metric        metric = 'chebyshev' for l_inifity, euclidean for l2
        metric = 'euclidean'
        features = data.x[0].cpu().numpy()
        db = DBSCAN(eps=0.9, min_samples=2,metric=metric).fit(features)
        db_labels = db.labels_
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        centroids = []
        for i in range(n_clusters):
            centroid = np.mean(features[db_labels==i], axis=0)
            centroids.append(centroid)
        distances = euclidean_distances(centroids)
        hier = AgglomerativeClustering(n_clusters=num_cuts, linkage='ward').fit(distances)
        hier_labels = hier.labels_
        for i in range(num_nodes):
            if db_labels[i]==-1:
                part = hier_labels[np.argmin(np.linalg.norm(features[i]-centroids,axis=1))]
            else:
                part=hier_labels[db_labels[i]]
            partitions[i][part]=1
            
    elif intial_type=='optics':
        metric = 'chebyshev'
        features = data.x[0].cpu()
        clust = OPTICS(min_samples=10,metric=metric).fit(features)
        clust_labels = clust.labels_
        n_clusters = len(set(clust_labels)) - (1 if -1 in clust_labels else 0)
        centroids = []
        for i in range(n_clusters):
            centroid = np.mean(features[clust_labels==i], axis=0)
            centroids.append(centroid)
        distances = euclidean_distances(centroids)
        hier = AgglomerativeClustering(n_clusters=num_cuts, linkage='ward').fit(distances)
        hier_labels = hier.labels_
        for i in range(num_nodes):
            if clust_labels[i]==-1:
                part = np.argmin(np.linalg.norm(features[i]-centroids,axis=1))
            else:
                part=hier_labels[clust_labels[i]]
            partitions[i][part]=1
    else:
        print("Initialisation type not supported") 
        exit(0)    
    return partitions

def get_order(data,initial_order):
    if initial_order=='random':
        # fix torch seed
        # torch.manual_seed(0)
        order = torch.randperm(data.num_nodes)
    elif initial_order=='core_value':
        order = data.x[1]
    elif initial_order=='degree':
        ## networkx graph from edge_index 
        G=to_networkx(Data(edge_index=data.edge_index,num_nodes=data.num_nodes),to_undirected=True)
        deg=nx.degree_centrality(G)
        order = torch.tensor(sorted(deg, key=deg.get, reverse=True))
    elif initial_order=='betweenness':
        G=to_networkx(Data(edge_index=data.edge_index,num_nodes=data.num_nodes),to_undirected=True)
        bet=nx.betweenness_centrality(G)
        order = torch.tensor(sorted(bet, key=bet.get, reverse=True))
    elif initial_order=='closeness':
        G=to_networkx(Data(edge_index=data.edge_index,num_nodes=data.num_nodes),to_undirected=True)
        clo=nx.closeness_centrality(G)
        order = torch.tensor(sorted(clo, key=clo.get, reverse=False))
    elif initial_order=='cluster':
        G=to_networkx(Data(edge_index=data.edge_index,num_nodes=data.num_nodes),to_undirected=True)
        clu=nx.clustering(G)
        order = torch.tensor(sorted(clu, key=clu.get, reverse=True))
    elif initial_order=='kcore':
        G=to_networkx(Data(edge_index=data.edge_index,num_nodes=data.num_nodes),to_undirected=True)
        cor=nx.core_number(G)
        order = torch.tensor(sorted(cor, key=cor.get, reverse=True))
    else:
        print("Initial order type not supported") 
        sys.exit(0)
    return order

def get_node_scores(data, partitions,node_select_heuristic,hops, device):
    Adj = torch.sparse_coo_tensor(data.edge_index.to(device), torch.ones(data.edge_index.shape[1]).to(device), (data.num_nodes, data.num_nodes)).to(device).double()
    Adj_pow = Adj
    A = Adj
    for i in range(hops-1):
        Adj_pow = torch.mm(Adj_pow, Adj)
        A += Adj_pow

    A = A.to_dense()
    A[A>0] = 1
 
    # calculate number of neighbours in same partition as node
    same_partition_map = torch.mm(partitions, partitions.t())
    same_partition_elementwise = same_partition_map * A.to_dense()

    same_partition_count = torch.sum(same_partition_elementwise, dim=1) 

    if node_select_heuristic=='diff':
        other_partition_count = torch.sum(A.to_dense(), dim=1) - same_partition_count

        node_scores = (other_partition_count+0.000001) / (same_partition_count+0.01)      # try subtract


    elif node_select_heuristic=='diff_max':
        node_partition = torch.argmax(partitions, dim=1)
        partition_scores = torch.mm(A, partitions)
        partition_scores.scatter_(1, node_partition.unsqueeze(1), 0)
        other_partition_count,_ = torch.max(partition_scores, dim=1) 

        node_scores = (other_partition_count+0.000001) / (same_partition_count+0.01)      # try subtract


    elif node_select_heuristic=='diff_max_scaled':
        node_partition = torch.argmax(partitions, dim=1)
        partition_scores = torch.mm(A, partitions)
        partition_scores.scatter_(1, node_partition.unsqueeze(1), 0)
        other_partition_count,_ = torch.max(partition_scores, dim=1)
        # multiply by normalised degree
        other_partition_count = other_partition_count * torch.sum(A.to_dense(), dim=1) / torch.sum(A.to_dense())

        node_scores = (other_partition_count+0.000001) / (same_partition_count+0.01)      # try subtract


    elif node_select_heuristic=='diff_max_balanced':
        node_partition = torch.argmax(partitions, dim=1)
        partition_scores = torch.mm(A, partitions)
        partition_scores.scatter_(1, node_partition.unsqueeze(1), 0)
        other_partition_count,_ = torch.max(partition_scores, dim=1) 

        node_scores = (other_partition_count+0.000001) / (same_partition_count+0.01)      # try subtract

        n = partitions.shape[0]
        oneT = torch.ones((1, n)).double().to(device)
        node_per_partition = torch.mm(oneT,partitions)
        balanced_scores = node_per_partition/n
        balanced_node_scores = balanced_scores[:,node_partition]

        node_scores = node_scores*balanced_node_scores

        # node_scores = node_scores.to(device)
        # A = A.to_sparse()
        # A = A.to(device)


    # breakpoint()
    node_scores=torch.tensor(node_scores)
    return node_scores

# def update_node_scores(node_scores, same_partition_count,curr_node_id,data,partitions,device):
#     neighbours = data.edge_index[1][data.edge_index[0]==curr_node_id]
#     # add curr_node to neighbours
#     neighbours = torch.cat((neighbours,torch.tensor([curr_node_id]).to(device)))
#     # update scores of neighbours

#     # only update scores of neighbours
#     same_partition_count
#     same_partition_neighbours = same_partition_count[neighbours]

def analyse_norm_cuts_SBM(dataset,type,result_folder,device):
    result_path_all=f"{result_folder}/{type}"
    
    for j in range(len(dataset)):
        data = dataset[j].to(device)
        result_path=f"{result_path_all}/{j+1}"
        # print(f"Gold NormCut {type}:",data.gold_norm_cut,f"Best NormCut {type}:",best_norm_cut)
        # print(f"Best NormCut {type}:",best_norm_cut)
        # print(f"Gold NormCut {type}:",data.hMetis_norm_cut)
        tot_nodes = data.num_nodes
        
        cuts = np.loadtxt(f"{result_path}/cuts.txt",dtype=int)
        cuts = torch.tensor(cuts).to(device)
        
        gt_cuts = torch.zeros(tot_nodes).to(device)
        # num cuts in every partition = tot_nodes/num_cuts
        # gt_cuts [0:tot_nodes/num_cuts] = maximum number in cuts[0:tot_nodes/num_cuts]
        for i in range(data.num_cuts):
            gt_cuts[i*int(tot_nodes/data.num_cuts):(i+1)*int(tot_nodes/data.num_cuts)] = torch.max(cuts[i*int(tot_nodes/data.num_cuts):(i+1)*int(tot_nodes/data.num_cuts)])
        # incorrect cuts -> indexes for which cuts!=gt_cuts
        incorrect_cuts = torch.where(cuts!=gt_cuts)[0]
        print("Analysing Dataset with type:",type,"and index:",j+1)
        print("Total Nodes:",tot_nodes)
        print("Total Incorrect cuts:",incorrect_cuts.numel())
        for node in incorrect_cuts:
            print("Node:",node.item())
            print("Correct cut:",gt_cuts[node].item())
            print("Predicted Cut:",cuts[node].item())
            #what are cuts of neighbours
            neighbours = data.edge_index[1][data.edge_index[0]==node]
            neighbour_cuts = torch.zeros((data.num_cuts,1)).to(device)
            for i in range(data.num_cuts):
                neighbour_cuts[i] = torch.sum(cuts[neighbours]==i)
            print("Neighbour cuts:",neighbour_cuts.squeeze().tolist())


def analyse_norm_cuts(dataset,type,result_folder,device):
    # result_folder = result_folder[:-5]      # TODO change this
    result_path_all=f"{result_folder}/{type}"
    
    for j in range(len(dataset)):
        data = dataset[j].to(device)
        # result_path=f"{result_path_all}/{j+1}"
        result_path=f"{result_path_all}"
        # print(f"Gold NormCut {type}:",data.gold_norm_cut,f"Best NormCut {type}:",best_norm_cut)
        # print(f"Best NormCut {type}:",best_norm_cut)
        # print(f"Gold NormCut {type}:",data.hMetis_norm_cut)
        tot_nodes = data.num_nodes
        # breakpoint()
        cuts = np.loadtxt(f"{result_path}/cuts.txt",dtype=int)
        cuts = torch.tensor(cuts).to(device)
        
        # find nodes whose neighbours are in different partition then self
        count=0
        for node in range(tot_nodes):
            neighbours = data.edge_index[1][data.edge_index[0]==node]
            neighbour_cuts = torch.zeros((data.num_cuts,1)).to(device)
            for i in range(data.num_cuts):
                neighbour_cuts[i] = torch.sum(cuts[neighbours]==i)
            if torch.sum(neighbour_cuts>0)>1 and cuts[node]!=torch.argmax(neighbour_cuts):
                print("Node:",node)
                print("Predicted Cut:",cuts[node].item())
                print("Neighbour cuts:",neighbour_cuts.squeeze().tolist())
                count+=1
        print("Number of confused Nodes:",count) 

def analyse_spectral(dataset,type,data_folder,device):
    result_path_all=f"{data_folder}/{type}"
    
    for j in range(len(dataset)):
        data = dataset[j].to(device)
        result_path=f"{result_path_all}/{j+1}"
        # print(f"Gold NormCut {type}:",data.gold_norm_cut,f"Best NormCut {type}:",best_norm_cut)
        # print(f"Best NormCut {type}:",best_norm_cut)
        # print(f"Gold NormCut {type}:",data.hMetis_norm_cut)
        tot_nodes = data.num_nodes
        
        cuts = np.loadtxt(f"{result_path}/cut_spectral.txt",dtype=int)
        cuts = torch.tensor(cuts).to(device)

        # find nodes whose neighbours are in different partition then self
        for node in range(tot_nodes):
            # breakpoint()
            neighbours = data.edge_index[1][data.edge_index[0]==node]
            neighbour_cuts = torch.zeros((data.num_cuts,1)).to(device)
            for i in range(data.num_cuts):
                neighbour_cuts[i] = torch.sum(cuts[neighbours]==i)
            if (torch.sum(neighbour_cuts>0)>1 and cuts[node]!=torch.argmax(neighbour_cuts)):
                print("Node:",node)
                print("Predicted Cut:",cuts[node].item())
                print("Neighbour cuts:",neighbour_cuts.squeeze().tolist())


