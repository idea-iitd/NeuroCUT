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
parser.add_argument('--num_perturbation',type=int,default=1, help='Number of perturbations for each node')
parser.add_argument('--wandb',type=str,default='false', help='wandb or not')
parser.add_argument('--scoring_func', type=str, default='None', help='scoring function to use (mlp or l1 or some other)')
parser.add_argument('--initial_type',type=str,default='random', help='random or kmeans initialization')
parser.add_argument('--initial_order',type=str,default='random', help='random or core_value initial order')
parser.add_argument('--node_select',type=str,default='false', help='select nodes or go according to order')
parser.add_argument('--node_select_heuristic',type=str,default='diff_max', help='which heuristic to use for node selection')
parser.add_argument('--pool',type=str,default='mean', help='which pool to get scores')
parser.add_argument('--update_reward',type=str,default='last', help='Reward Computation')
parser.add_argument('--hops',type=int,default=2, help='Num hops for score and heuristic computation')
parser.add_argument('--cuttype',type=str,default='normalised', help='Type of cut to optimise')
parser.add_argument('--finetune_epochs',type=int,default=0, help='Total fine tuning epochs')



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
cuttype=args.cuttype
finetune_epochs=args.finetune_epochs


if (gpu > 3 or gpu < 0): 
    print("Not a Valid gpu ")
    exit(0)
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
print(device)

# result_folder=f'../results/split_{args.train}_{args.embedding}_{args.norm}'
result_folder=f'../results/{args.train}_{args.embedding}_{args.anchors}_{args.norm}_{model_name}_{num_layers}_{args.modelScore}_{args.iter_update_params}_{args.initial_type}_{args.initial_order}_{args.scoring_func}_{args.node_select}_{num_perturbation}_{args.node_select_heuristic}_{args.pool}_{args.update_reward}_{args.hops}_{args.cuttype}_finetune_{args.finetune_epochs}'

print("Result Folder: ", result_folder)
core_node_path=f'../results_phase1/{args.train}_{args.embedding}_{args.norm}_{model_name}_{num_layers}'

if wandb_flag:
    wandb.init(project="phase2_rl", name=f"{args.train}_{args.embedding}_{args.anchors}_{args.norm}_{model_name}_{num_layers}_{args.modelScore}_{args.iter_update_params}_{args.initial_type}_{args.initial_order}_{args.scoring_func}_{args.node_select}_{num_perturbation}_{args.node_select_heuristic}_{args.pool}_{args.update_reward}_{args.hops}_{args.cuttype}_finetune_{args.finetune_epochs}", config=args,entity='mincuts')

model_folder=f'../models/{args.train}_{args.embedding}_{args.anchors}_{args.norm}_{model_name}_{num_layers}_{args.modelScore}_{args.iter_update_params}_{args.initial_type}__{args.initial_order}_{args.scoring_func}_{args.node_select}_{num_perturbation}_{args.node_select_heuristic}_{args.pool}_{args.update_reward}_{args.hops}_{args.cuttype}_finetune_{args.finetune_epochs}'

# if (not os.path.exists(model_folder) ):
#     need_training=True

os.makedirs(result_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(result_folder+"/train_set", exist_ok=True)
os.makedirs(result_folder+"/val_set", exist_ok=True)
os.makedirs(result_folder+"/test_set",exist_ok=True)
# print(result_path)
# using dataset class defined in datasets.py
train_dataset=dataset(dataset_path+'/train_set',embedding,norm,core_node_path+'/train_set',device,args=args) 
val_dataset=dataset(dataset_path+'/val_set',embedding,norm,core_node_path+'/val_set',device,args=args)
test_dataset=dataset(dataset_path+'/test_set',embedding,norm,core_node_path+'/test_set',device,args=args)
# print(len(train_dataset[0].x))
# DataLoader helps in batching
dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=True)

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
if wandb_flag:
    wandb.watch(model)

# breakpoint()
    
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay=5e-6)
criterion = torch.nn.CrossEntropyLoss()            # TODO loss function : crossEntropy loss
model=model.double()
if need_training:
    torch.save(model.state_dict(), model_folder+'/bestval.pth')
    print("Saving Initial Mode(bestval)")
best_loss = -1
bestmodel = None

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
        _ , embeddings = model(node_feat, data.edge_index, torch.tensor(0), partitions,pool, data.node_weights) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
        for i in range(num_nodes):
            # curr_node_id = int(data.x[1][i].item())            # data.x[1] is the node_ordering 
            if node_select:
                # sample from node_scores
                curr_node_id = torch.tensor(torch.multinomial(node_scores,1).item())
            else:
                curr_node_id = order[i%num_nodes]  ## get the node id from the order
            partitions[curr_node_id] = torch.zeros((1,data.num_cuts)).to(device)
        
            out, _ = model(node_feat, data.edge_index, curr_node_id, partitions,pool,data.node_weights,embeddings) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
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

def train(epoch,data_loader):  

    model.train()
    running_loss = 0.0
    optimizer.zero_grad()  # Clear gradients.

    total_reward=0

    # ## CHANGING
    # norm_cutsss=[]

    print("Epoch ############################",epoch)
    for _,data in enumerate(data_loader):
        num_nodes = data.num_nodes
        node_feat = data.x[0].clone() 
        # print("done?")
        partitions = intialise_assgiment(data,initial_type, device)
        norm_cut=getNormalisedCutValue(data,partitions,data.num_cuts ,device,cuttype).item()
        if math.isnan(norm_cut):
            print("ERORR")
            exit(0)
        num_partitions = data.num_cuts          # TODO how to make this generic
        rewards=[]
        actions=[]
        logits=[]
        prev_cut=norm_cut       # TODO : make this best cut
        init_cut=prev_cut
        last_cut=prev_cut
        if node_select:
            node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)
        else:
            order = get_order(data,initial_order)

        for i in tqdm(range(num_nodes*num_perturbation)):
            # if (i==2):
            #     break
            # curr_node_id = int(data.x[1][i].item())            # data.x[1] is the node_ordering 
            # if i%num_nodes==0:
            #     cuts_new = [-1 for ii in range(num_nodes)]
            #     for ii in range(num_nodes):
            #         cuts_new[ii] = torch.argmax(partitions[ii]).item()
            #     result_path=f"{result_folder}/train_set"
            #     np.savetxt(f"{result_path}/cuts.txt",np.array(cuts_new), fmt='%d')
            #     # breakpoint()
            #     analyse_norm_cuts(train_dataset,'train_set',result_folder,device)
            #     print("Done for i:",i)
            #     print("###############")
                

            if node_select:
                # print(node_scores)
                curr_node_id = torch.tensor(torch.multinomial(node_scores,1).item())
            else:
                curr_node_id = order[i%num_nodes]
            ## remove curr_node_id from partitions

            if update_reward=='teacher_force_nan':
                old_partitions = partitions.clone()

            partitions[curr_node_id] = torch.zeros((1,num_partitions)).to(device)
            out,_ = model(node_feat, data.edge_index, curr_node_id, partitions,pool, data.node_weights) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
                        # features, edge_index, node_id, partitions, core_values
            # print("out",out,out.shape)  ## out is of shape [1,num_cuts]
            out=out.reshape((1,data.num_cuts))
            prob=out.squeeze(dim=0)     ## out is not of shape [num_cuts]
            ## Pick actions based on out
            prob = torch.softmax(prob,dim=0)  ## normalize out
            a = torch.multinomial(prob,1).item()
            partitions = partitions.clone()
            partitions[curr_node_id][a] = 1
            new_norm_cut = getNormalisedCutValue(data, partitions, data.num_cuts ,device,cuttype,default=3*init_cut).item()
            if update_reward=='teacher_force_nan':
                if abs(new_norm_cut-3*init_cut)<1e-5:
                    print("NAN")
                    partitions = old_partitions

            reward = get_reward(prev_cut,new_norm_cut)
            total_reward+=reward
            if node_select:
                node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)

            ## just for printing
            last_cut=new_norm_cut

            # ## CHANGING
            # norm_cutsss.append(new_norm_cut)

            ## TODO : update prev_cut
            if update_reward=='best':
                prev_cut = min(prev_cut,new_norm_cut)
            elif update_reward=='last':
                prev_cut = new_norm_cut
            elif update_reward=='last_non_nan' or update_reward=='teacher_force_nan':
                if not (abs(new_norm_cut-3*init_cut)<1e-5):
                    prev_cut = new_norm_cut
            else:
                print("What is this update reward?")
                exit(0) 
            ## APPEND TO REWARDS, ACTIONS, LOGITS
            rewards.append(reward)
            actions.append(a)
            logits.append(out.squeeze(dim=0))
            
            # print("rewards",rewards)
            # print("actions",actions)
            # print("logits",logits)
            if (i%args.iter_update_params==0 and i!=0) or  i==num_nodes*num_perturbation-1: ## TODO : do we need to update at end node
                ## Calculate cummulative reward
                
                cum_reward=[0 for j in range(len(rewards))]
                reward_len = len(rewards)
                for j in reversed(range(reward_len)):
                    cum_reward[j] = rewards[j] + (cum_reward[j+1]*gamma if j+1 < reward_len else 0)
                a=torch.tensor(actions,dtype=torch.int64).to(device)
                l=torch.stack(logits)
                cum_reward=torch.tensor(cum_reward,dtype=torch.float).to(device)

                ## Calculate loss
                log_probs = -F.cross_entropy(l, a, reduction="none")
                loss = -log_probs * cum_reward
                loss.sum().backward()
                optimizer.step()
                ## Clear rewards, actions, logits, grads
                optimizer.zero_grad()
                rewards=[]
                actions=[]
                logits=[]

                # cum_loss=0

        if variance and epoch%10==0:
            save_variance(model,data,device,epoch)

    # plt.clf()
    # plt.plot([i for i in range(len(norm_cutsss))],norm_cutsss,label="Normalised cuts")
    # plt.legend()
    # plt.title("Normalized cut vs Perturbation for training")
    # plt.xlabel("Perturbation")
    # plt.ylabel("Normalized cut")
    # plt.draw()
    # result_path=f"{result_folder}/train_set"
    # plt.savefig(f"{result_path}/Ncut_{epoch}.png")
    # plt.close()

    total_reward/=len(train_dataset)
    print('Epoch %d, total_reward: %.5f,init_cut: %.5f,final_cut: %.05f' % (epoch + 1, total_reward,init_cut,last_cut))
    return total_reward,init_cut,last_cut

# bestValCut = -1
bestValNormCut = -1
norm_cut_values=[]   # list for checking what is happening at test time
def getPartitions_rl(model,data,device,iter_update_params=5,store_norm_cuts=False):
    
    with torch.no_grad():
        model.eval()           
        num_nodes = data.num_nodes
        node_feat = data.x[0].clone()           
        partitions = intialise_assgiment(data,initial_type,device)
        # temp_paritions=partitions.copy()
        actions=[]
        logits=[]
        init_cut = getNormalisedCutValue(data,partitions,data.num_cuts ,device,cuttype).item()
        if node_select:
            node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)
        else:
            order = get_order(data,initial_order)

        if store_norm_cuts:
            norm_cut=getNormalisedCutValue(data,partitions,data.num_cuts ,device,cuttype).item()
            norm_cut_values.append(norm_cut)

        # get the transformed node_embeddings
        _ , embeddings = model(node_feat, data.edge_index, torch.tensor(0), partitions,pool, data.node_weights) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
        
        for i in range(num_nodes*num_perturbation):
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
                
            ## remove curr_node_id from partitions
            partitions[curr_node_id] = torch.zeros((1,data.num_cuts)).to(device)
            # print("curr_node_id",curr_node_id)
            # print("partitions",partitions)    

            out, _ = model(node_feat, data.edge_index, curr_node_id, partitions,pool,data.node_weights,embeddings) # TODO make this generic      # partitions is list of nodes?(only the correct partition?) # make this shape (1,num_cuts)
                        # features, edge_index, node_id, partitions, core_values
            # print("out",out,out.shape)  ## out is of shape [1,num_cuts]
            out=out.reshape((1,data.num_cuts))
            prob=out.squeeze(dim=0)     ## out is not of shape [num_cuts]
            ## Pick actions based on out
            prob = torch.softmax(prob,dim=0)  ## normalize out
            # print("prob",prob,prob.shape)
            a = torch.argmax(prob).item()
            partitions[curr_node_id][a] = 1
            ## TODO: calculate reward

            if node_select:
                node_scores = get_node_scores(data,partitions,node_select_heuristic,hops,device)

            if store_norm_cuts:
                norm_cut = getNormalisedCutValue(data, partitions, data.num_cuts ,device,cuttype,default=3*init_cut).item()
                norm_cut_values.append(norm_cut)
                # print(i)

            ## APPEND TO REWARDS, ACTIONS, LOGITS
            actions.append(a)
            logits.append(out.squeeze(dim=0))
            # print("rewards",rewards)
            # print("actions",actions)
            # print("logits",logits)
            # # exit(0)
            if (i%iter_update_params==0 and i!=0) or  i==num_nodes*num_perturbation-1: ## TODO : do we need to update at end node
                actions=[]
                logits=[]
        last_cut = getNormalisedCutValue(data, partitions, data.num_cuts ,device,cuttype,default=3*init_cut).item()
    return partitions,last_cut

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
            partitions,norm_cut = getPartitions_rl(model=model, data=data, device=device, iter_update_params=args.iter_update_params)
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
            print("Saving Model Val")
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
            partitions,norm_cut = getPartitions_rl(model=model, data=data, device=device, iter_update_params=args.iter_update_params,store_norm_cuts=True)
            if (best_norm_cut==-1 or norm_cut < best_norm_cut) and (not math.isnan(norm_cut)):
                best_norm_cut = norm_cut
                best_partitions = partitions
                best_norm_cuts=norm_cut_values.copy()
        
        if wandb_flag:
            print("here")
            wandb.log({f"hMetis NormCut {type}":data.hMetis_norm_cut,f"spectral NormCut {type}":data.spectral_norm_cut,f"Best NormCut {type}":best_norm_cut})
        # print(f"Gold NormCut {type}:",data.gold_norm_cut,f"Best NormCut {type}:",best_norm_cut)
        # print(f"Best NormCut {type}:",best_norm_cut)
        # print(f"Gold NormCut {type}:",data.hMetis_norm_cut)
        tot_nodes = data.num_nodes
        cuts = [-1 for i in range(tot_nodes)]
        # best_partitions is shape (num_nodes, num_parts)
        for i in range(tot_nodes):
            cuts[i] = torch.argmax(best_partitions[i]).item()       # TODO check variance here
        # for part_id in best_partitions:
        #     for node_id in best_partitions[part_id]:
        #         cuts[node_id] = part_id
        # with open(f"{result_path}/cuts.txt",'w') as f:
        #     sys.stdout=f
        #     print(cuts)

        np.savetxt(f"{result_path}/cuts.txt",np.array(cuts), fmt='%d')

        print("Best Norm Cut Test:", best_norm_cut)

        orgininal_std=sys.stdout
        with open(f"{result_path}/stats.txt",'w') as f:
            sys.stdout=f
            # for i in range(len(best_partitions)):
            #     print("Partition ", i, " : ", len(best_partitions[i]))
            # cut_value_train = getCutValue(data, best_partitions, device)
            # print("Cut Value: ", cut_value_train
            print("Normalised Cut Value: ", best_norm_cut)
        sys.stdout=orgininal_std

        plot_cuts(dataset_path+f'/{type}/{j+1}/graph.txt',cuts, result_path)
        
        plt.clf()
        plt.plot([i for i in range(len(best_norm_cuts))],best_norm_cuts,label="Normalised cuts")
        plt.legend()
        plt.title("Normalized cut vs Perturbation")
        plt.xlabel("Perturbation")
        plt.ylabel("Normalized cut")
        plt.draw()
        plt.savefig(f"{result_path}/NcutsVsPert.png")
        plt.close()
    
        print(f"Done {type} {j+1}")

if __name__ == '__main__':

    rewards=[]
    rewards_test=[]
    init_cut_list = []
    init_cut_list_test = []
    last_cut_list = []
    last_cut_list_test = []
    val_cut_list = []
    val_norm_cut_list = []
    val_norm_cut_list_test = []

    if (need_training):
        print("Started Training")
        for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times
            # print('epoch ', epoch + 1)
            reward,init_cut,last_cut= train(epoch,dataloader)
            rewards.append(reward)
            init_cut_list.append(init_cut)
            last_cut_list.append(last_cut)
            val_norm_cut = test(val_dataset)
            # val_cut_list.append(val_cut)
            val_norm_cut_list.append(val_norm_cut)

            print(f"Train Reward:{reward},Val Norm Cut:{val_norm_cut},Init Cut:{init_cut},Last Cut:{last_cut}")
        ## saving model
            if wandb_flag:
                wandb.log({"Train Reward":reward,"Val Norm Cut":val_norm_cut,"Init Cut":init_cut,"Last Cut":last_cut},step=epoch)
        # if wandb_flag:
        #     wandb.log({'line_plot': wandb.plot.line_series(
        #     xs=np.array([[i for i in range(len(init_cut_list))], [i for i in range(len(init_cut_list))]]), ys=np.array([init_cut_list, last_cut_list]),
        #     series=["init_cut", "last_cut"], title="Change in Norm cut")})
        torch.save(model.state_dict(), model_folder+'/lastmodel.pth')

        model.load_state_dict(torch.load(model_folder+'/bestval.pth'))
        model.eval()
        print("Test Set Stats")
        calculate_stats(test_dataset,"test_set")

        print("Started FineTuning")
        for epoch in tqdm(range(finetune_epochs)):  # loop over the dataset multiple times
            # print('epoch ', epoch + 1)
            reward_test,init_cut_test,last_cut_test= train(epoch,test_dataloader)
            rewards_test.append(reward_test)
            init_cut_list_test.append(init_cut_test)
            last_cut_list_test.append(last_cut_test)
            val_norm_cut_test = test(val_dataset)
            # val_cut_list.append(val_cut)
            val_norm_cut_list_test.append(val_norm_cut)
        ## saving model
            print(f"Train Reward Test:{reward_test},Val Norm Cut Test:{val_norm_cut_test},Init Cut Test:{init_cut_test},Last Cut Test:{last_cut_test}")
            if wandb_flag:
                print("logging finetune")
                wandb.log({"Train Reward Test":reward_test,"Val Norm Cut Test":val_norm_cut_test,"Init Cut Test":init_cut_test,"Last Cut Test":last_cut_test},step=epoch)
        # if wandb_flag:
        #     wandb.log({'line_plot': wandb.plot.line_series(
        #     xs=np.array([[i for i in range(len(init_cut_list))], [i for i in range(len(init_cut_list))]]), ys=np.array([init_cut_list, last_cut_list]),
        #     series=["init_cut", "last_cut"], title="Change in Norm cut")})
        torch.save(model.state_dict(), model_folder+'/lastmodel.pth')
        # print('Performing test')
        # test(test_dataset,test=True)      # on test set
        # print('Finished Training')

        average_reward = []
        for idx in range(len(rewards)):
            avg_list = np.empty(shape=(1,), dtype=int)
            if idx < 50:
                avg_list = rewards[:idx+1]
            else:
                avg_list = rewards[idx-49:idx+1]
            average_reward.append(np.average(avg_list))
        
        # print(rewards)
        # print(average_reward)
        # print(val_cut_list)
        # print(val_norm_cut_list)
        # print(init_cut_list)
        # print(last_cut_list)

        if variance:
            plot_variance()

        plt.clf()
        plt.plot([i for i in range(len(rewards))],rewards,label="Train_Rewards")
        plt.plot([i for i in range(len(average_reward))],average_reward,label="Average Reward")
        plt.legend()
        plt.title("Reward vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.draw()
        plt.savefig(f"{result_folder}/reward.png")
        plt.close()

        plt.clf()
        plt.plot([i for i in range(len(init_cut_list))],init_cut_list,label="init_cut",linestyle='dashed', linewidth = 3,marker='o', markerfacecolor='blue', markersize=5)
        plt.plot([i for i in range(len(last_cut_list))],last_cut_list,label="last_cut",linestyle='dashed', linewidth = 3,marker='o', markerfacecolor='green', markersize=5)
        plt.legend()
        plt.title("change vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("change")
        plt.draw()
        plt.savefig(f"{result_folder}/change.png")
        plt.close()

        

        # plt.clf()
        # plt.plot([i for i in range(len(val_cut_list))],val_cut_list,label="Val_Cut")
        # plt.legend()
        # plt.title("Val_cut vs Epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Val_Cut")
        # plt.draw()
        # plt.savefig(f"{result_folder}/cut.png")
        # plt.close()


        plt.clf()
        plt.plot([i for i in range(len(val_norm_cut_list))],val_norm_cut_list,label="Val_norm_Cut")
        plt.legend()
        plt.title("Val_norm_cut vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Val_Norm_Cut")
        plt.draw()
        plt.savefig(f"{result_folder}/norm_cut.png")
        plt.close()


    print("Calculating Stats")

    # analyse_norm_cuts(train_dataset,'train_set',result_folder,device)
    # analyse_norm_cuts(val_dataset,'val_set',result_folder,device)
    # exit(0)

    print("model folder: ", model_folder)


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

    
