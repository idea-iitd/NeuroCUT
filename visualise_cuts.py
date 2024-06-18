## takes input of folder containing graph.txt and cuts.txt
## redistribute nodes to surface and core nodes
## save the node type in a input folder as cuts.pt
## visualize the cuts using visualize_cuts.png
## save the image in the same folder

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D
import re
import json
import math
import os

def getcutvalue(cuts,num_cuts,A,cuttype,node_weights):
    # Y = partitions
    # get sparse adjecency matrix from edge_index
    # A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]), (data.num_nodes, data.num_nodes))
    # A=sparse_mx_to_torch_sparse_tensor(nx.adjacency_matrix(to_networkx(data)))           # TODO make this again?
    Y = torch.zeros((cuts.shape[0],num_cuts))          # TODO : change num_cuts to g
    Y[torch.arange(cuts.shape[0]), cuts] = 1
    # print(Y)
    if cuttype=='normalised':
        D = torch.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())        # time taking
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        normalised_cut = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense())
        ## TODO : check this 
        # print(default)
    elif cuttype=='kmin':
        Y_t = (1 - Y).t()
        k_cut = torch.sum(torch.mm(Y, Y_t) * A.to_dense())
        num_edge=torch.sum(A)
        normalised_cut=k_cut/num_edge
    elif cuttype=='sparsest':
        n=A.shape[0]
        count=torch.sum(Y,axis=0)
        gamma=torch.min(n-count,count) ## min |S|,|SBAR|
        Ybygamma=torch.div(Y,gamma) 
        Y_t = (1 - Y).t()
        k_cut = torch.sum(torch.mm(Ybygamma, Y_t) * A.to_dense())
        normalised_cut=k_cut
    elif cuttype=='sparsest_weight':
        weights=(node_weights*Y.t()).t()
        weight_count=torch.sum(weights,axis=0)
        total_weight=torch.sum(weight_count)
        gamma=torch.min(total_weight-weight_count,weight_count) ## min |S|,|SBAR|
        print(gamma)
        Ybygamma=torch.div(Y,gamma) 
        Y_t = (1 - Y).t()
        Y_t=Y_t.double()
        k_cut = torch.sum(torch.mm(Ybygamma, Y_t) * A.to_dense())
        normalised_cut=k_cut

    elif cuttype=='GAP':
        D = torch.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        l1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense())
        oneT = torch.ones((1, Y.shape[0]))
    # # Loss2 is normalizing term
        n = Y.shape[0]
        # l2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t()))
        l2 = torch.sum((torch.mm(oneT,Y) - n/num_cuts)**2)
        normalised_cut = l1+l2
    elif cuttype=='GAP_1':
        D = torch.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        l1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense())
        oneT = torch.ones((1, Y.shape[0]))
    # # Loss2 is normalizing term
        n = Y.shape[0]
        # l2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t()))
        l2 = torch.sum((torch.mm(oneT,Y) - n/num_cuts)**2)
        l2 = l2/(n**2)
        normalised_cut = l1+l2
    elif cuttype=='GAP_2':
        D = torch.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()
        l1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense())
        oneT = torch.ones((1, Y.shape[0]))
    # # Loss2 is normalizing term
        n = Y.shape[0]
        node_per_partition = torch.mm(oneT,Y)
        node_per_partition = node_per_partition/n
        product = torch.prod(node_per_partition)
        # l2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t()))
        l2 = product
        normalised_cut = l1-l2
    elif cuttype=='conductance':
        D = torch.sum(A, dim=1).to_dense()
        tot_edge = torch.sum(D)
        gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
        Ybygamma=torch.div(Y,gamma.t()) 
        Y_t = (1 - Y).t()
        phi_node = torch.sum(torch.mm(Ybygamma, Y_t) * A.to_dense(),axis=0)
        phi = torch.mm(Y.t(),phi_node.float().unsqueeze(1))
        ind = torch.nonzero(gamma.squeeze() < tot_edge.item()/2).squeeze()
        phi_ind = phi[ind]
        normalised_cut = torch.min(phi_ind)
    else:
        print("Wrong type")
        exit(0)
    return normalised_cut

def normalised_cuts(cuts, A,num_cuts=5):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    # Y is n*k matrix where n is number of nodes and k is number of partitions
    # make Y from cuts by one hot encoding
    Y = torch.zeros((cuts.shape[0],num_cuts))          # TODO : change num_cuts to g
    Y[torch.arange(cuts.shape[0]), cuts] = 1
    n = Y.shape[0]
    g = Y.shape[1]
    D = torch.sum(A, dim=1)
    # print(A.shape)
    # print(cuts.shape)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense())
    return loss

def edge_cut(g,cuts):
    cut_value = 0
    for e in g.edges():
        v1 = e[0]
        v2 = e[1]
        if cuts[v1]!=cuts[v2]:
            cut_value += 1
    return cut_value

def distribute(g, cuts):
    # 0 -> surface nodes
    # 1 -> core nodes
    node_type = {i:1 for i in g.nodes()}
    for edge in g.edges():
        if (edge[0] in list(g.nodes)) and (edge[1] in list(g.nodes)) and cuts[edge[0]]!=cuts[edge[1]]:
            # print(edge)
            node_type[edge[0]] = 0
            node_type[edge[1]] = 0
    return node_type

def read_num_cuts(file):
    f=open(file)
    content=f.read()
    num=re.findall(r'\d+',content)
    return int(num[0])

def wrapper2(g,num_cuts,cuts,folder):
    A = nx.adjacency_matrix(g)
    A = torch.from_numpy(A.toarray())
    norm_cut=normalised_cuts(cuts,A,num_cuts=num_cuts)
    d={"gold_norm_cut":norm_cut.item()}
    print(norm_cut.item())
    with open(folder+'gold_norm_cut.txt', "w") as fp:
        json.dump(d, fp)  # encode dict into JSON

def wrapper(g,cuts,num_cuts,suffix,make_plots=True):
    #check number of arguments
    A = nx.adjacency_matrix(g)
    A = torch.from_numpy(A.toarray())
    norm_cut=normalised_cuts(cuts,A,num_cuts=num_cuts)
    # cut_value=edge_cut(g,cuts)
    colour={0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'maroon', 5:'cyan', 6:'magenta', 7:'black', 8:'orange', 9:'purple'}
    if make_plots:
        plt.clf()
        node_color=[]
        for node in g.nodes():
            node_color.append(colour[cuts[node].item()])
        
        nx.draw(g, node_color=node_color,node_size=20, with_labels=False)
        legend_elements = []
        for i in range(num_cuts):
            legend_elements.append( Line2D([0], [0], marker='o', color=colour[i], label=f'cuts_{i}'))
        plt.legend(handles=legend_elements, loc='upper right')
        
        # nx.draw(g, node_color=node_color, cmap=plt.cm.Set1, with_labels=True)

        plt.savefig(folder+f"/visualize_cuts_{suffix}.png")

    # node_type = distribute(g, cuts)
    # l=[]
    # cut=[]
    # core=0
    # surface=0
    # for node in g.nodes():
    #     l.append(colour[node_type[node]])
    #     if node_type[node]==1:  
    #         core+=1
    #     else:   
    #         surface+=1
    
    # for node in range(len(list(g.nodes()))):
    #     cut.append(node_type[node])
    
    # # file1 = open(folder+'/stats.txt', "a")  # append mode
    # # file1.write(f"Normalised cut value for {suffix}= {norm_cut}\n")
    # # file1.write(f"cut value for {suffix}= {cut_value}\n")
    # # file1.write(f"BASELINE: {suffix}:: Core nodes: {core}, Surface nodes: {surface}\n")   
    # # file1.close()
    # if (suffix=='hMetis'):
    #     torch.save(cut, folder+f'/core_surface.pt')
    # torch.save(cut, folder+f'/core_surface_{suffix}.pt')
    # plt.clf()
    ## core nodes -> blue, surface nodes -> red,
    # nx.draw(g, node_color=l, with_labels=True)
    # if (len(list(g.nodes())) < 101):
    #     nx.draw(g, node_color=l, cmap=plt.cm.Set1, with_labels=True,node_size=20)
    # else:
    #     nx.draw(g, node_color=l, cmap=plt.cm.Set1, with_labels=False,node_size=20)

    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='blue', label='core'),
    #     Line2D([0], [0], marker='o', color='red', label='surface'),        
    # ]
    # plt.legend(handles=legend_elements, loc='upper right')
    
    # plt.savefig(folder+f"/visualize_nodes_{suffix}.png")
    return norm_cut

def read_num_nodes(folder):
    with open(f'{folder}graph_stats.txt') as f:
        data = f.read()
    js = json.loads(data)
    return js['num_nodes'],js['num_cuts']

def dump_all_metrics(folder,num_cuts,A):
    if os.path.exists(folder+'node_weights.txt'):
        node_weights=torch.from_numpy(np.loadtxt(folder+'node_weights.txt'))
        cuts_hMetis_weights = torch.from_numpy(np.loadtxt(folder+'cut_hMetis_weights.txt')).long()
    else:
        node_weights=None
    
    cuts_hMetis = torch.from_numpy(np.loadtxt(folder+'cut_hMetis.txt')).long()
    cuts_spect = torch.from_numpy(np.loadtxt(folder+'cut_spectral.txt')).long()
    cuts_gap = torch.from_numpy(np.loadtxt(folder+'cut_gap.txt')).long()

    cut_types = ['normalised','kmin','sparsest','sparsest_weight','GAP','GAP_1','GAP_2','conductance']
    # cut_types = ['conductance']
    # cut_types = ['GAP_1','GAP_2']
    hMetis_metrics = {}
    for cut_type in cut_types:
        if cut_type == 'sparsest_weight':
            if node_weights==None:
                continue
            cut_val = getcutvalue(cuts_hMetis_weights,num_cuts,A,cut_type,node_weights)
            hMetis_metrics[cut_type] = cut_val.item()
        else:
            cut_val = getcutvalue(cuts_hMetis,num_cuts,A,cut_type,node_weights)
            hMetis_metrics[cut_type] = cut_val.item()
    
    
    with open(folder+'hMetis_metrics.json', "w") as json_file:
        # Dump the dictionary into the JSON file
        json.dump(hMetis_metrics, json_file)

    
    spectral_metrics = {}
    for cut_type in cut_types:
        if cut_type == 'sparsest_weight' and node_weights==None:
            continue
        cut_val = getcutvalue(cuts_spect,num_cuts,A,cut_type,node_weights)
        spectral_metrics[cut_type] = cut_val.item()
    
    with open(folder+'spectral_metrics.json', "w") as json_file:
        # Dump the dictionary into the JSON file
        json.dump(spectral_metrics, json_file)

    gap_metrics = {}
    for cut_type in cut_types:
        if cut_type == 'sparsest_weight' and node_weights==None:
            continue
        # breakpoint()
        cut_val = getcutvalue(cuts_gap,num_cuts,A,cut_type,node_weights)
        gap_metrics[cut_type] = cut_val.item()
    
    with open(folder+'gap_metrics.json', "w") as json_file:
        # Dump the dictionary into the JSON file
        json.dump(gap_metrics, json_file)


if __name__ == '__main__':


    # G=nx.Graph()
    # G.add_nodes_from([i for i in range(5)])
    # G.add_edges_from([[0,1],[0,2],[0,4],[1,2],[1,4],[2,3],[3,4]])
    # nodes_weights=torch.tensor([1,2,3,4,5])
    # A = nx.adjacency_matrix(G)
    # A = torch.from_numpy(A.toarray())

    # cuts=torch.tensor([0,1,1,2,0])
    # print(getcutvalue(cuts,3,A,'sparsest_weight',nodes_weights))
    # exit(0)

    if len(sys.argv) != 2:
        print('Usage: python visualize_cuts.py <folder>')
        sys.exit(1)


    folder=sys.argv[1]
    graph_file=folder+'graph.txt'
    num_nodes,num_cuts=read_num_nodes(folder)

    tgraph=nx.read_edgelist(graph_file, nodetype=int)
    G=nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(tgraph.edges())
    A = nx.adjacency_matrix(G)
    A = torch.from_numpy(A.toarray())
    A = A.double()

    dump_all_metrics(folder,num_cuts,A)


    # node_weights=torch.from_numpy(np.loadtxt(folder+'node_weights.txt'))
    # cuts_hMetis = torch.from_numpy(np.loadtxt(folder+'cut_hMetis_weights.txt'))
    # print("hMetis done")
    # cuts_spect = torch.from_numpy(np.loadtxt(folder+'cuts_spectral_35.txt'))
    # print("Spectal done")
    # cuts_hMetis=cuts_hMetis.long()
    # cuts_spect=cuts_spect.long()

    # wrapper(G,cuts_spect,5,'spectral')
    # print("sparsest_weight for hMetis",getcutvalue(cuts_hMetis,num_cuts,A,'sparsest_weight',node_weights))
    # print("sparsest_weight for Spectral",getcutvalue(cuts_spect,num_cuts,A,'sparsest_weight',node_weights))


    # cuts_temp = torch.from_numpy(np.loadtxt(folder+'cut_temp.txt'))
    # print("temp done")
    
    # cuts_temp=cuts_temp.long()
    
    # print("sparsest_weight for temp",getcutvalue(cuts_temp,num_cuts,A,'sparsest_weight',node_weights))


    # print("sparsestcut for hMetis",getcutvalue(cuts_hMetis,num_cuts,A,'sparsest'))
    # print("sparsestcut for Spectral",getcutvalue(cuts_spect,num_cuts,A,'sparsest'))

    # # print(num_cuts)

    # breakpoint()
    # TODO get cuts_hMetis and cuts_spect first
    
    # norm_hMetis=wrapper(G,cuts_hMetis,num_cuts,"hMetis",False).item()
    # norm_spect=wrapper(G,cuts_spect,num_cuts,"spectral",False).item()
    # print(norm_hMetis)
    # print(norm_spect)
    # cuts_spect = torch.from_numpy(np.loadtxt(folder+'cuts_spectral_35.txt'))
    # cuts_spect=cuts_spect.long()
    # norm_spect=wrapper(G,cuts_spect,num_cuts,"spectral35",False).item()
    # with open(f'{folder}graph_stats.txt') as f:
    #     data = f.read()
    # js = json.loads(data)
    # js['hMetis_norm_cut']=norm_hMetis
    # # js['spectral_norm_cut']=norm_spect
    # js['spectral_norm_cut_35']=norm_spect
    # print(js)
    # with open(f'{folder}graph_stats.txt', "w") as fp:
    #     json.dump(js, fp)
    # wrapper(g,cuts,5,"hMetis")


    # cuts = torch.from_numpy(np.loadtxt(folder+'cut_gomory.txt'))
    # cuts=cuts.long()
    # wrapper(g,cuts,num_cuts,"gomory")

    # cuts = torch.from_numpy(np.loadtxt(folder+'cut.txt'))
    # cuts=cuts.long()
    # wrapper(g,cuts,num_cuts,"hMetis")

    # cuts = torch.from_numpy(np.loadtxt(folder+'cut_karger.txt'))
    # cuts=cuts.long()
    # wrapper(g,cuts,num_cuts,"karger")

