export train,embedding,anchors,epochs,need_training,model_name,trained_on,norm,modelScore,iter_update_params,isAssigned_feature,wandb,num_run,initial_type,initial_order,scoring_func,num_perturbation,node_select,node_select_heuristic,pool,update_reward,hops,num_perturbation_inference,cuttype,finetune_epochs, gamma
train = cora_lc
embedding = given_lipchitz ##coefficents, Lipschitz_rw, Lipschitz_sp
anchors = 35			
epochs = 3
need_training=True
model_name=graphsage      ## gcn,gat,graphsage
num_layers=2
trained_on=gap
norm=percentile      # MinMax, Standard, Normalizer,percentile, None
modelScore = ModelLinkPred 			## ModelBasic, ModelAtten, ModelLinkPred, ModelLocalLinkPred
iter_update_params = 2   ## for iteratiing rl parameters
isAssigned_feature = false
gpu=2
wandb=false   ## wandb on of true or false
num_run=5 ## for running multiple times in val and test
initial_type=kmeans_Linf   ## random,kmeans
scoring_func=mlp     # mlp,	l1, l2, cosine, dot
initial_order=random  ## random, core_value
num_perturbation=1
node_select=true		# true or false
node_select_heuristic=diff_max_balanced		# diff, diff_max, diff_max_scale_degree
pool=mean ## mean, max, sum
update_reward=last ## last,best
hops=2  # 2,3,..
num_perturbation_inference=1000
cuttype=normalised  ##normalised,kmin,sparsest
finetune_epochs=0
gamma=0.99

main:
	python main.py --cuttype $(cuttype) --update_reward $(update_reward) --pool $(pool) --wandb $(wandb) --num_perturbation $(num_perturbation) --num_run $(num_run) --initial_order $(initial_order) --initial_type $(initial_type) --train $(train) --gpu $(gpu) --embedding $(embedding) --anchors $(anchors) --epochs $(epochs) --model_name $(model_name) --num_layers $(num_layers) --need_training $(need_training)  --norm $(norm) --modelScore $(modelScore) --iter_update_params $(iter_update_params) --isAssigned_feature $(isAssigned_feature) --scoring_func $(scoring_func) --node_select $(node_select) --node_select_heuristic $(node_select_heuristic) --hops $(hops) --gamma $(gamma)

finetune:
	python finetune.py --cuttype $(cuttype) --update_reward $(update_reward) --pool $(pool) --wandb $(wandb) --num_perturbation $(num_perturbation) --num_run $(num_run) --initial_order $(initial_order) --initial_type $(initial_type) --train $(train) --gpu $(gpu) --embedding $(embedding) --anchors $(anchors) --epochs $(epochs) --model_name $(model_name) --num_layers $(num_layers) --need_training $(need_training)  --norm $(norm) --modelScore $(modelScore) --iter_update_params $(iter_update_params) --isAssigned_feature $(isAssigned_feature) --scoring_func $(scoring_func) --node_select $(node_select) --node_select_heuristic $(node_select_heuristic) --hops $(hops) --gamma $(gamma) --finetune_epochs $(finetune_epochs)


clean_pkl: 
	find ../data -name \*.pkl -type f -delete
