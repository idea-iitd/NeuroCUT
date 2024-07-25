train = cora_lc
embedding = given_lipchitz##coefficents,  Lipschitz_rw, Lipschitz_sp,Lipschitz_rw_only,spectral, Lipschitz_rw_node_weights
anchors = 35# default 20 for 500, 3 for 30
epochs = 40
need_training=true		# true or false
model_name=graphsage      ## gcn,gat,graphsage,mlp
num_layers=2
trained_on=gap
norm=percentile      # MinMax, Standard, Normalizer, percentile,None
modelScore = ModelLinkPred 			## ModelBasic, ModelAtten, ModelLinkPred, ModelLocalLinkPred
iter_update_params = 2
isAssigned_feature = false
gpu=0                         ##0,1,2,3
wandb=false   ## wandb on of true or false
num_run=1 ## for running multiple times in val and test
initial_type=kmeans_Linf   ## random,kmeans,kmeans_Linf,kmeansSpectral
scoring_func=mlp     # mlp,	l1, l2, cosine, dot
initial_order=random  ## random, degree,closeness,betweenness,core_value,cluster,kcore
num_perturbation=2
node_select=true
node_select_heuristic=diff_max_balanced		# diff, diff_max, diff_max_scaled, diff_max_balanced
pool=mean ## mean, max, sum
update_reward=last ## last,best,last_non_nan,teacher_force_nan
hops=2  # 2,3,..
num_perturbation_inference=1000
cuttype=normalised  ##normalised,kmin,sparsest,sparsest_weight
finetune_epochs=0
gamma=0.99
# threshold=0.5
# pos_weight=0.3
# num_times_samples=50
# num_edges_samples=50
## Add below line to incorporate above variables
# pos_weight=$(pos_weight) threshold=$(threshold) num_edges_samples=$(num_edges_samples) num_times_samples=$(num_times_samples)

main:
	$(MAKE) -C src cuttype=$(cuttype) update_reward=$(update_reward) pool=$(pool) wandb=$(wandb) num_perturbation=$(num_perturbation) num_run=$(num_run) initial_order=$(initial_order) initial_type=$(initial_type) train=$(train) gpu=$(gpu) embedding=$(embedding) anchors=$(anchors) epochs=$(epochs) model_name=$(model_name) num_layers=$(num_layers) need_training=$(need_training)  norm=$(norm) modelScore=$(modelScore) iter_update_params=$(iter_update_params) isAssigned_feature=$(isAssigned_feature) scoring_func=$(scoring_func) node_select=$(node_select) node_select_heuristic=$(node_select_heuristic) hops=$(hops) gamma=$(gamma)
inference: 
	$(MAKE) -C src num_perturbation_inference=$(num_perturbation_inference)  update_reward=$(update_reward) pool=$(pool) wandb=$(wandb) num_perturbation=$(num_perturbation) num_run=$(num_run) initial_order=$(initial_order) initial_type=$(initial_type) train=$(train) gpu=$(gpu) embedding=$(embedding) anchors=$(anchors) epochs=$(epochs) model_name=$(model_name) num_layers=$(num_layers) need_training=$(need_training)  norm=$(norm) modelScore=$(modelScore) iter_update_params=$(iter_update_params) isAssigned_feature=$(isAssigned_feature) scoring_func=$(scoring_func) node_select=$(node_select) node_select_heuristic=$(node_select_heuristic) hops=$(hops) inference
finetune:
	$(MAKE) -C src cuttype=$(cuttype) update_reward=$(update_reward) pool=$(pool) wandb=$(wandb) num_perturbation=$(num_perturbation) num_run=$(num_run) initial_order=$(initial_order) initial_type=$(initial_type) train=$(train) gpu=$(gpu) embedding=$(embedding) anchors=$(anchors) epochs=$(epochs) model_name=$(model_name) num_layers=$(num_layers) need_training=$(need_training)  norm=$(norm) modelScore=$(modelScore) iter_update_params=$(iter_update_params) isAssigned_feature=$(isAssigned_feature) scoring_func=$(scoring_func) node_select=$(node_select) node_select_heuristic=$(node_select_heuristic) hops=$(hops) finetune_epochs=$(finetune_epochs) finetune

clean_pkl: 
	find ./data -name \*.pkl -type f -delete