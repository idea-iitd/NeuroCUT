import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
from utils import *
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from scipy import stats
import torch.nn.functional as F
import re
import json
from torch_geometric.utils import from_networkx


class dataset(Dataset):
    def __init__(self,folder,embeding,norm,beta,device='cpu',args=None):
        super().__init__()
        self.folder=folder
        self.embeding=embeding
        self.anchors=args.anchors
        self.device=device
        self.norm=norm
        self.isAssigned_feature=args.isAssigned_feature
        self.beta = beta
        # TODO make generic 
        self.num_no_features=None            # change when we add more node features
        # if isAssigned_feature=='true':
        #     self.num_no_features+=1
        self.num_ed_features=1            # change when we add more edge features
        self.generate_features(1)       
        print("Total Node Features : ", self.num_no_features)

    def read_num_nodes(self,index):
        with open(self.folder+f'/{index}/graph_stats.txt') as f:
            data = f.read()
        js = json.loads(data)
        num_nodes=js['num_nodes']
        return num_nodes
    
    def generate_edge_index(self,index):
        edge_list_path = self.folder+f'/{index}/graph.txt'
        if self.embeding=='given_lipchitz' or self.embeding=='given_spectral' or self.embeding=='given_lipchitz_beta':
            options = ['cora_lc','citeseer_lc','harbin','roman_empire','actor','ogbn_arxiv','coauthor','questions','horse']
            for g in options:
                if self.folder.startswith(g):
                    edge_list_path=f'../raw_data/{g}/graph.txt'
                    break
            # TODO check it

        num_nodes=self.read_num_nodes(index)
        tgraph=nx.read_edgelist(edge_list_path, nodetype=int)
        graph=nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(tgraph.edges())

        d = from_networkx(graph)
        return d.edge_index
        # A = nx.adjacency_matrix(self.graph)
        # A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        # norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        # adj = sparse_mx_to_torch_sparse_tensor(norm_adj) # SciPy to Torch sparse
        # As = sparse_mx_to_torch_sparse_tensor(A)
    
    def generate_features(self,index):
        options = ['cora_lc','citeseer_lc','harbin','roman_empire','actor','ogbn_arxiv','coauthor','questions','horse']
        if (self.embeding=="coefficents"):
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.norm}.pkl'
        elif (self.embeding=='spectral'):
            filename=self.folder+f'/{index}/features_spectral_{self.anchors}_{self.norm}.pkl'
        elif (self.embeding=='given'):
            for g in options:
                if g in self.folder:
                    filename=f'../raw_data/{g}/node_embeddings.pt'
                    break
        elif (self.embeding=='given_spectral'):
            filename = self.folder+f'/{index}/features_{self.embeding}_{self.anchors}_{self.norm}.pkl'
            for g in options:
                if g in self.folder:
                    f2=f'../raw_data/{g}/node_embeddings.pt'
                    break
        
        elif (self.embeding=='given_lipchitz'):
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.anchors}_{self.norm}.pkl'
            for g in options:
                if g in self.folder:
                    f2=f'../raw_data/{g}/node_embeddings.pt'
                    break

        elif (self.embeding=='given_lipchitz_weight'):
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.anchors}_{self.norm}.pkl'
            for g in options:
                if g in self.folder:
                    f2=f'../raw_data/{g}/node_embeddings.pt'
                    break

        elif (self.embeding=='given_lipchitz_beta'):
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.anchors}_{self.norm}_{self.beta}.pkl'
            for g in options:
                if g in self.folder:
                    f2=f'../raw_data/{g}/node_embeddings.pt'
                    break 

            
        else:
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.anchors}_{self.norm}.pkl'
        # print(filename)
        if os.path.exists(filename):      # TODO (we are everytime generating new features)

            # print("Loading from pickle file")
            features = torch.load(open(filename,"rb"))
            features = features.double()

            self.num_no_features=features.shape[1]  # set the total num of node features
        else:
            # tgraph=nx.read_edgelist(self.folder+f'/{index}/graph.txt')
            # if type(list(tgraph.nodes())[0])==str:
            #     mapping= {i:int(i) for i in tgraph.nodes()}
            #     tgraph=nx.relabel_nodes(tgraph,mapping)
            
            # l=sorted(tgraph.nodes())
            # graph=nx.Graph()
            # graph.add_nodes_from(l)
            # graph.add_edges_from(tgraph.edges())
            num_nodes=self.read_num_nodes(index)
            tgraph=nx.read_edgelist(self.folder+f'/{index}/graph.txt', nodetype=int)
            graph=nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(tgraph.edges())
            # print(graph.nodes())
            # print(graph.edges())
            # exit(0)             
            # features=[list(cluster_centrality(graph).values()),list(degree_centrality(graph).values()),list(betweenness_centrality(graph).values()),list(closeness_centrality(graph).values())]
            # features=[list(cluster_centrality(graph).values()),list(closeness_centrality(graph).values())]

            features_list=["cluster_centrality","degree_centrality","betweenness_centrality","closeness_centrality","kcore"] 

            features = []
            if self.embeding!='Lipschitz_rw_only' and self.embeding!='Lipschitz_rw_node_weights' and self.embeding!='spectral' and self.embeding!='given_lipchitz' and self.embeding!='given_lipchitz_beta' and self.embeding!='given_lipchitz_weight' :
                for feature_name in features_list:
                    if feature_name=='cluster_centrality':
                        feat=cluster_centrality(graph)
                    elif feature_name=='closeness_centrality':
                        feat=closeness_centrality(graph)
                    elif feature_name=='degree_centrality':
                        feat=degree_centrality(graph)
                    elif feature_name=='betweenness_centrality':
                        feat=betweenness_centrality(graph)
                    elif feature_name=='kcore':
                        feat=core_number(graph)
                    feat = [feat[i] for i in range(len(feat))]
                    features.append(feat)
                
                features=torch.tensor(features).t()     ### features (num_nodes,3)
            if self.isAssigned_feature.lower()=='true':
                # make features (num_nodes,num_cuts)
                print("features before",features.shape)
                features=torch.cat((features,torch.tensor([0]*len(graph.nodes())).unsqueeze(1)),1)
                # print(features.shape)
                print("features",features.shape)        ### features ( num_nodes, 4 )
            # print("shape after transpose", features.shape)   
            if self.embeding=='spectral':
                spe=spectral_embedding(graph,self.anchors,len(graph.nodes()))
                features=torch.from_numpy(spe)


            if self.embeding=='Lipschitz_rw_only' or self.embeding=='Lipschitz_rw_node_weights':
                
                lip=lipschitz_rw_embedding(graph,self.anchors,len(graph.nodes()),self.beta)
                features=lip
                # sum=torch.sum(features,dim=0)
                # features=features/sum
            if self.embeding=='Lipschitz_rw_weight_only':
                
                lip=lipschitz_rw_embedding_weight(graph,self.anchors,len(graph.nodes()),self.beta)
                features=lip

            if (self.embeding=='Lipschitz_rw'):
                lip=lipschitz_rw_embedding(graph,self.anchors,len(graph.nodes()))
                features=torch.cat((features,lip),1)
                # sum=torch.sum(features,dim=0)
                # features=features/sum

            if (self.embeding=='Lipschitz_sp'):
                ## TODO add code for sp
                lip=lipschitz_embedding(graph,self.anchors,len(graph.nodes()))
                features=torch.cat((features,lip),1)  
                # sum=torch.sum(features,dim=0)
                # features=features/sum

            if (self.embeding=='pca'):
                pca=pca_embedding(graph,self.anchors)
                features=torch.cat((features,pca),1) 
            
            if self.embeding=='given_lipchitz' or self.embeding=='given_lipchitz_beta':
                lip=lipschitz_rw_embedding(graph,self.anchors,len(graph.nodes()),self.beta)
                features=lip

                features_given = torch.load(open(f2,"rb"))
                features_given = features_given.double()

                features = torch.cat((features, features_given), dim=1)

            if self.embeding=='given_lipchitz_weight':
                lip=lipschitz_rw_embedding_weight(graph,self.anchors,len(graph.nodes()),self.beta)
                features=lip

                features_given = torch.load(open(f2,"rb"))
                features_given = features_given.double()

                features = torch.cat((features, features_given), dim=1)


            if self.embeding=='given_spectral':
                spe=spectral_embedding(graph,self.anchors,len(graph.nodes()))
                features=torch.from_numpy(spe)

                features_given = torch.load(open(f2,"rb"))
                features_given = features_given.double()

                features = torch.cat((features, features_given), dim=1)


                # minn=torch.min(features,dim=0)
                # sum=torch.sum(features,dim=0)
                # features=(features+minn.values)/sum

            if self.embeding=='Lipschitz_rw_node_weights':
                node_weights=torch.from_numpy(np.loadtxt(self.folder+f'/{index}/node_weights.txt'))
                tot_nodes = len(node_weights)
                node_weights = torch.tensor(node_weights).reshape(tot_nodes,1)
                node_weights = node_weights.repeat(1,self.anchors)
                features = torch.cat((features, node_weights), dim=1)
                print("node weights added in features")
            print("feature generated")

            features=features.type(torch.DoubleTensor)
            self.num_no_features=features.shape[1]  # set the total num of node features

            if self.norm=="MinMax":
                scaler=MinMaxScaler()
                features=torch.tensor(scaler.fit_transform(features))
            elif self.norm == "Standard":
                scaler=StandardScaler()
                features=torch.tensor(scaler.fit_transform(features))
            elif self.norm == "Normalizer":
                scaler=Normalizer()
                features=torch.tensor(scaler.fit_transform(features))
            elif self.norm == "percentile":
                features=torch.tensor(features).t()
                temp=[(stats.rankdata(f, 'average')-1)/len(f) for f in features]
                features=torch.tensor(temp).t()
            elif self.norm!="None":         # use this in case of percentile
                print("Norm not present")
                exit(0)
                        
            with open(filename, "wb") as f:
                torch.save(features, f)
            # print(type(features))
        return features

    def read_num_cuts(self,index):
        with open(self.folder+f'/{index}/graph_stats.txt') as f:
            data = f.read()
        js = json.loads(data)
        num_cuts=js['num_cuts']
        return num_cuts
        
    def read_hMetis_norm_cut(self,index):
        with open(self.folder+f'/{index}/graph_stats.txt') as f:
            data = f.read()
        js = json.loads(data)
        if 'hMetis_norm_cut' not in js:
            return -1
        return js['hMetis_norm_cut']
    
    def read_spectral_norm_cut(self,index):
        with open(self.folder+f'/{index}/graph_stats.txt') as f:
            data = f.read()
        js = json.loads(data)
        if 'spectral_norm_cut' not in js:
            return -1
        return js['spectral_norm_cut']
  
    def generate_y(self,index):

        cuts=torch.from_numpy(np.loadtxt(self.folder+f'/{index}/cut.txt'))

        cuts=cuts.long()
        cuts=cuts.type(torch.DoubleTensor)
        return cuts

    def len(self):
        return len(os.listdir(self.folder))
    
    def get(self, idx):
        # print(idx)
        # x=[self.generate_features(idx+1), node_ordering, core_values]
        x=[self.generate_features(idx+1)]
        # print(x[0].shape)
        # exit(0)
        edge_index=self.generate_edge_index(idx+1)
        #print(edge_index)
        num_cuts=self.read_num_cuts(idx+1)
        hmetis_norm_cut=self.read_hMetis_norm_cut(idx+1)
        spectral_norm_cut=self.read_spectral_norm_cut(idx+1)

        node_weights=torch.tensor([1 for i in range(x[0].shape[0])])
        if self.embeding=='Lipschitz_rw_node_weights':
            node_weights=torch.from_numpy(np.loadtxt(self.folder+f'/{idx+1}/node_weights.txt'))
        return Data(x=x,edge_index=edge_index,hMetis_norm_cut=hmetis_norm_cut,spectral_norm_cut=spectral_norm_cut,num_nodes=x[0].shape[0],num_cuts=num_cuts,node_weights=node_weights).to(self.device)

