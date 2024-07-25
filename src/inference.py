import torch
import os
from model import ModelBasic, ModelLinkPred, ModelAtten, ModelLocalLinkPred
from dataset import dataset
import numpy
from utils import *
import networkx as nx
from torch_geometric.loader import DataLoader
import argparse
import sys
from tqdm import tqdm
from model_utils import *
import copy
import wandb
from torch.nn import functional as F
import time
import json

def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Random seed set as {}".format(seed))

set_seed()


parser = argparse.ArgumentParser(
    description="Command Line Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--embedding', type=str, default='coefficients',
                    help='What embedding to use (Lipschitz_rw or Lipschitz_sp or coefficients )')
parser.add_argument('--train', type=str, default='cora',
                help='Dataset to train')
parser.add_argument('--anchors', type=int, default=30, help='Number of Anchors for Lipschitz')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs train')
parser.add_argument('--model_name',type=str,default='gcn', help='Use whether gcnconv or gatconv')
parser.add_argument('--need_training',type=str,default='false', help='training or testing')
parser.add_argument('--norm',type=str,default='none', help='Use whether none,MinMax,Standard or Normalizer')
parser.add_argument('--modelScore',type=str,default='none', help='Use ModelGCN or ModelLinkPred or ModelLocalLinkPred or ModelGCN_Atten')
parser.add_argument('--num_layers',type=int,default=3, help='Number of Layers')
parser.add_argument('--iter_update_params',type=int,default=0, help='Total Iterations after Loss is Backproped')
parser.add_argument('--isAssigned_feature',type=str,default='false', help='Keep a feature which tells if the node is assigned or not')
parser.add_argument('--gpu', type=int, default=4, help='gpu to train')
parser.add_argument('--num_run',type=int,default=10, help='Number of runs for val and test')
# parser.add_argument('--num_perturbation',type=int,default=1, help='Number of perturbations for each node')
parser.add_argument('--num_perturbation',type=float,default=1, help='Number of perturbations for each node')      # TODO earlier it was int
parser.add_argument('--wandb',type=str,default='false', help='wandb or not')
parser.add_argument('--scoring_func', type=str, default='None', help='scoring function to use (mlp or l1 or some other)')
parser.add_argument('--initial_type',type=str,default='random', help='random or kmeans initialization')
parser.add_argument('--initial_order',type=str,default='random', help='random or core_value initial order')
parser.add_argument('--node_select',type=str,default='false', help='select nodes or go according to order')
parser.add_argument('--node_select_heuristic',type=str,default='diff_max', help='which heuristic to use for node selection')
parser.add_argument('--pool',type=str,default='mean', help='which pool to get scores')
parser.add_argument('--update_reward',type=str,default='last', help='Reward Computation')
parser.add_argument('--hops',type=int,default=2, help='Num hops for score and heuristic computation')
parser.add_argument('--num_perturbation_inference',type=int,default=1000, help='Num perturbations for inference')
parser.add_argument('--cuttype',type=str,default='normalised', help='Type of cut to optimise')
parser.add_argument('--gamma',type=float,default=0.99, help='value of gamma for discounted rewards')
parser.add_argument('--alpha_reward',type=float,default=0, help='value of alpha for modularity reward scaling')
parser.add_argument('--beta',type=int,default=100, help='value of beta for RWR total iterations')


# parser.add_argument('--training', type=bool,default=False)
args = parser.parse_args()
# print(args)
# exit(0)
# Variables of Arg Parser
num_epochs=args.epochs
dataset_path='../data/'+args.train
num_layers = args.num_layers
embedding=args.embedding
anchors=args.anchors
model_name=args.model_name
temp=args.need_training.lower()
need_training = temp=='true'
norm=args.norm
gamma=0.99 ##discount factor TODO: Make it a parameter to be tuned by RL 
temp=args.wandb.lower()
wandb_flag = temp=='true'
num_run=args.num_run
initial_type=args.initial_type
scoring_func=args.scoring_func
initial_order=args.initial_order
node_select=(args.node_select.lower()=='true')
node_select_heuristic=args.node_select_heuristic
num_perturbation=args.num_perturbation
pool = args.pool
gpu=args.gpu
update_reward=args.update_reward.lower()
hops=args.hops
num_perturbation_inference=args.num_perturbation_inference  ## TODO : make parameter


# print(update_reward)
if (gpu > 3 or gpu < 0): 
    print("Not a Valid gpu ")
    exit(0)
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
print(device)


# result_folder=f'../results/split_{args.train}_{args.embedding}_{args.norm}'
result_folder=f'../results/{args.train}_{args.embedding}_{args.anchors}_{args.norm}_{model_name}_{num_layers}_{args.modelScore}_{args.iter_update_params}_{args.initial_type}_{args.initial_order}_{args.scoring_func}_{args.node_select}_{num_perturbation}_{args.node_select_heuristic}_{args.pool}_{args.update_reward}_{args.hops}_{args.cuttype}_{args.gamma}'
core_node_path=f'../results_phase1/{args.train}_{args.embedding}_{args.norm}_{model_name}_{num_layers}'


model_folder=f'../models/{args.train}_{args.embedding}_{args.anchors}_{args.norm}_{model_name}_{num_layers}_{args.modelScore}_{args.iter_update_params}_{args.initial_type}__{args.initial_order}_{args.scoring_func}_{args.node_select}_{num_perturbation}_{args.node_select_heuristic}_{args.pool}_{args.update_reward}_{args.hops}_{args.cuttype}_{args.gamma}_{args.alpha_reward}_{args.beta}'

# if (not os.path.exists(model_folder) ):
#     need_training=True

# print(result_path)
# using dataset class defined in datasets.py
train_dataset=dataset(args.train,dataset_path+'/train_set',embedding,norm,core_node_path+'/train_set',device,args=args) 
val_dataset=dataset(args.train,dataset_path+'/val_set',embedding,norm,core_node_path+'/val_set',device,args=args)
test_dataset=dataset(args.train,dataset_path+'/test_set',embedding,norm,core_node_path+'/test_set',device,args=args)
# print(len(train_dataset[0].x))

# DataLoader helps in batching
dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)

if args.modelScore=='ModelBasic':
    model = ModelBasic(num_features=train_dataset.num_no_features, num_layers = num_layers, model_name = model_name, hidden_channels=32,device=device).to(device)   # Note change num_features when needed
elif args.modelScore=='ModelLinkPred':
    model = ModelLinkPred(num_features=train_dataset.num_no_features, num_layers = num_layers, model_name = model_name, hidden_channels=32,scoring_func=scoring_func,device=device).to(device)   # Note change num_features when needed
elif args.modelScore=='ModelAtten':
    model = ModelAtten(num_features=train_dataset.num_no_features, num_layers = num_layers, model_name = model_name, hidden_channels=32,device=device).to(device)   # Note change num_features when needed
elif args.modelScore=='ModelLocalLinkPred':
    model = ModelLocalLinkPred(num_features=train_dataset.num_no_features, num_layers = num_layers, model_name = model_name, hidden_channels=32, hops=hops ,scoring_func=scoring_func,device=device).to(device)   # Note change num_features when needed
else:
    print("What Model is this?")
    exit(0)

    

model=model.double()

torch.autograd.set_detect_anomaly(True)

## for forget
# forget_losses=[]
# findex=[]
def get_reward(prev_cut,new_norm_cut):
    # TODO changing reward
    # reward = ((prev_cut - new_norm_cut)*100)/(abs(prev_cut)+abs(new_norm_cut)+0.1)
    reward = ((prev_cut - new_norm_cut)*100)/(prev_cut+new_norm_cut)
    return reward
variance=False
variances=[]
variances_epochs=[]

def save_variance(model,data,device,epoch=0):
    var=[]
    with torch.no_grad():
        model.eval()           
        num_nodes = data.num_nodes
        node_feat = data.x[0].clone()           
        partitions = intialise_assgiment(data,initial_type,device)

        if node_select:
            node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)
        else:
            order = get_order(data,initial_order)

        # get the transformed node_embeddings
        _ , embeddings = model(node_feat, data.edge_index, torch.tensor(0), partitions,pool) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
        for i in range(num_nodes):
            # curr_node_id = int(data.x[1][i].item())            # data.x[1] is the node_ordering 
            if node_select:
                # sample from node_scores
                curr_node_id = torch.tensor(torch.multinomial(node_scores,1).item())
            else:
                curr_node_id = order[i%num_nodes]  ## get the node id from the order
            partitions[curr_node_id] = torch.zeros((1,data.num_cuts)).to(device)
        
            out, _ = model(node_feat, data.edge_index, curr_node_id, partitions,pool,embeddings) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
            out=out.reshape((1,data.num_cuts))
            prob=out.squeeze(dim=0)     ## out is not of shape [num_cuts]
            ## Pick actions based on out
            prob = torch.softmax(prob,dim=0)  ## normalize out
            # print("prob",prob,prob.shape)

            a = torch.multinomial(prob,1).item()
            partitions[curr_node_id][a] = 1

            if node_select:
                node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)

            # calculte variance of prob
            var.append(torch.var(prob).item())
    variances.append(var)
    variances_epochs.append(epoch)

def plot_variance():
    plt.clf()
    for i in variances:
        plt.plot([j for j in range(len(i))],i,label=f"Epoch: {variances_epochs[variances.index(i)]}")
    plt.xlabel("Node id")
    plt.ylabel("Variance")
    plt.legend()
    plt.savefig(f"{result_folder}/variances.png")

# bestValCut = -1
bestValNormCut = -1
norm_cut_values=[]   # list for checking what is happening at test time

pertubation_info = []

def getPartitions_rl(model,data,device,iter_update_params=5,store_norm_cuts=False):

    pertubation_info_dummy = []


    start=time.time()
    with torch.no_grad():
        model.eval()           
        num_nodes = data.num_nodes
        node_feat = data.x[0].clone()           
        partitions = intialise_assgiment(data,'kmeans',device)
        # temp_paritions=partitions.copy()
        actions=[]
        logits=[]
        init_cut = getNormalisedCutValue(data,partitions,data.num_cuts ,device,args.cuttype).item()
        pertubation_info_dummy.append(init_cut)
        if node_select:
            node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)
        else:
            order = get_order(data,initial_order)

        # get the transformed node_embeddings
        _ , embeddings = model(node_feat, data.edge_index, torch.tensor(0), partitions,pool, data.node_weights) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)

        for i in tqdm(range(int(num_nodes*num_perturbation))):
            # curr_node_id = int(data.x[1][i].item())            # data.x[1] is the node_ordering 
            if node_select:
                # sample from node_scores
                curr_node_id = torch.tensor(torch.multinomial(node_scores,1).item())
            else:
                curr_node_id = order[i%num_nodes]  ## get the node id from the order
            # if curr_node_id==189:
            #     cuts = [-1 for i in range(data.num_nodes)]
            #     # best_partitions is shape (num_nodes, num_parts)
            #     for i in range(data.num_nodes):
            #         cuts[i] = torch.argmax(partitions[i]).item()
            #     cuts = torch.tensor(cuts).to(device)
            #     neighbours = data.edge_index[1][data.edge_index[0]==189]
            #     neighbour_cuts = torch.zeros((data.num_cuts,1)).to(device)
            #     for i in range(data.num_cuts):
            #         neighbour_cuts[i] = torch.sum(cuts[neighbours]==i)
            #     breakpoint()
            # print(curr_node_id)

            old_partition_id = torch.argmax(partitions[curr_node_id]).item()

            info_curr = [curr_node_id.item(),old_partition_id]
                
            ## remove curr_node_id from partitions
            partitions[curr_node_id] = torch.zeros((1,data.num_cuts)).to(device)
            # print("curr_node_id",curr_node_id)
            # print("partitions",partitions)    

            out, _ = model(node_feat, data.edge_index, curr_node_id, partitions,pool,data.node_weights,embeddings) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
                        # features, edge_index, node_id, partitions, core_values
            # print("out",out,out.shape)  ## out is of shape [1,num_cuts]
            out=out.reshape((1,data.num_cuts))
            prob=out.squeeze(dim=0)     ## out is not of shape [num_cuts]
            prob_scores_to_save = prob
            ## Pick actions based on out
            prob = torch.softmax(prob,dim=0)  ## normalize out
            # print("prob",prob,prob.shape)
            a = torch.argmax(prob).item()
            partitions[curr_node_id][a] = 1


            last_cut = getNormalisedCutValue(data, partitions, data.num_cuts ,device,args.cuttype,default=3*init_cut).item()


            info_curr.append(a)
            info_curr.append(last_cut)
            info_curr.append(prob_scores_to_save.tolist())
            ## TODO: calculate reward

            if node_select:
                node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)
                # print(i)
            
            pertubation_info_dummy.append(info_curr)

    ed=time.time()
    print("Time taken",ed-start)
    last_cut = getNormalisedCutValue(data, partitions, data.num_cuts ,device,args.cuttype,default=3*init_cut).item()
    print("Last Cut",last_cut)
    return partitions,last_cut,pertubation_info_dummy

## Test 
## set test false for validation
def test(test_set,test=False):         
    global bestValNormCut
    global bestmodel
    
    test_cut = 0
    test_norm_cut = 0
    for j in range(len(test_set)):
        best_norm_cut=-1
        best_partitions=None
        tp=None
        for _ in range(num_run): ## TODO how many runs
            data = test_set[j]
            if test:
                print(f"Test set {j},iterations: {_}",end=" ")
            else:
                print(f"Val set {j},iterations: {_}",end=" ")
            partitions,norm_cut,pertubation_info_dummy = getPartitions_rl(model=model, data=data, device=device, iter_update_params=args.iter_update_params)
            tp=partitions
            if (best_norm_cut==-1 or norm_cut < best_norm_cut) and (not math.isnan(norm_cut)):
                best_norm_cut = norm_cut
                best_partitions = partitions    
        test_cut += getCutValue(data, best_partitions,device)
        test_norm_cut += best_norm_cut
    # TODO change this for test vs val these
    # test_cut = test_cut/len(test_set)
    test_norm_cut = test_norm_cut/len(test_set)
    if not test:
        if (bestValNormCut==-1 or test_norm_cut < bestValNormCut) and  (not math.isnan(test_norm_cut)):
            bestValNormCut = test_norm_cut
            torch.save(model.state_dict(), model_folder+'/bestval.pth')
            bestmodel = model    
    if test:
        print('Test Cut : %.5f, Test Norm Cut : %.5f' % (test_cut, test_norm_cut))
    else:
        print('Val Cut : %.5f, Val Norm Cut : %.5f' % (test_cut, test_norm_cut))
    return test_norm_cut   


# Calculate stats for dataset given
# Dataset : train_dataset,val_dataset,test_dataset
# type: train_set,val_set,test_set
def calculate_stats(dataset,type):   
    result_path_all=f"{result_folder}/{type}"

    
    for j in range(len(dataset)):
        data = dataset[j].to(device)
        result_path=f"{result_path_all}/{j+1}"

        os.makedirs(result_path,exist_ok=True)
        best_norm_cut=-1
        best_partitions=None
        best_norm_cuts=None
        for _ in range(num_run): ## TODO how many runs
            norm_cut_values.clear()
            partitions,norm_cut,pertubation_info_dummy = getPartitions_rl(model=model, data=data, device=device, iter_update_params=args.iter_update_params,store_norm_cuts=True)
            if (best_norm_cut==-1 or norm_cut < best_norm_cut) and (not math.isnan(norm_cut)):
                best_norm_cut = norm_cut
                best_partitions = partitions
                best_norm_cuts=norm_cut_values.copy()
                pertubation_info = pertubation_info_dummy
        # print(f"Gold NormCut {type}:",data.gold_norm_cut,f"Best NormCut {type}:",best_norm_cut)
        # print(f"Best NormCut {type}:",best_norm_cut)
        # print(f"Gold NormCut {type}:",data.hMetis_norm_cut)
        tot_nodes = data.num_nodes
        cuts = [-1 for i in range(tot_nodes)]
        # best_partitions is shape (num_nodes, num_parts)
        for i in range(tot_nodes):
            # print(best_partitions[i])       #TODO analyse if model is very sure or just guessing
            cuts[i] = torch.argmax(best_partitions[i]).item()
        
        print(f"{type} Norm Cut:", best_norm_cut)

        # Write the list to a file
        with open(f'/DATATWO/users/mincut/BTP-Final/perturbation_results/{args.train}_{args.cuttype}_{args.node_select}.txt', 'w') as f:
            for sublist in pertubation_info:
                f.write(json.dumps(sublist) + "\n")
        # for part_id in best_partitions:
        #     for node_id in best_partitions[part_id]:
        #         cuts[node_id] = part_id
        # with open(f"{result_path}/cuts.txt",'w') as f:
        #     sys.stdout=f
        #     print(cuts)

        # np.savetxt(f"{result_path}/cuts.txt",np.array(cuts), fmt='%d')

        # orgininal_std=sys.stdout
        # with open(f"{result_path}/stats.txt",'w') as f:
        #     sys.stdout=f
        #     # for i in range(len(best_partitions)):
        #     #     print("Partition ", i, " : ", len(best_partitions[i]))
        #     # cut_value_train = getCutValue(data, best_partitions, device)
        #     # print("Cut Value: ", cut_value_train
        #     print("Normalised Cut Value: ", best_norm_cut)
        # sys.stdout=orgininal_std

        # plot_cuts(dataset_path+f'/{type}/{j+1}/graph.txt',cuts, result_path)
        
        # plt.clf()
        # plt.plot([i for i in range(len(best_norm_cuts))],best_norm_cuts,label="Normalised cuts")
        # plt.legend()
        # plt.title("Normalized cut vs Perturbation")
        # plt.xlabel("Perturbation")
        # plt.ylabel("Normalized cut")
        # plt.draw()
        # plt.savefig(f"{result_path}/NcutsVsPert.png")
        # plt.close()
    
        # print(f"Done {type} {j+1}")

if __name__ == '__main__':

    print("Calculating Stats")

    # analyse_norm_cuts(train_dataset,'train_set',result_folder,device)
    # analyse_norm_cuts(val_dataset,'val_set',result_folder,device)
    # exit(0)

    print("model_folder: ", model_folder)


    # some metrics to calculate stats on the test set   

    # train set
    print("Train Set Stats")
    model.load_state_dict(torch.load(model_folder+'/lastmodel.pth'))
    model.eval()

    calculate_stats(train_dataset,"train_set")

    
    model.load_state_dict(torch.load(model_folder+'/bestval.pth'))
    model.eval()

    print("Val Set Stats")
    calculate_stats(val_dataset,"val_set")

    print("Test Set Stats")
    calculate_stats(test_dataset,"test_set")

    # analyse_norm_cuts(test_dataset,'test_set',result_folder,device)

    # analyse spectral
    # analyse_spectral(train_dataset,'train_set',dataset_path,device)

    
