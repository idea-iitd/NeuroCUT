# NeuroCUT: A Neural Approach for Robust Graph Partitioning

This repository is an official implementation of the Paper : [NeuroCUT: A Neural Approach for Robust Graph Partitioning](https://arxiv.org/abs/2310.11787), accepted in KDD'24.


## Commands to run
1. Activate Environment or create from [requirements.txt](./requirements.txt)
```
source {environment name}/bin/activate
```
2. Various parameters are listed in [makefile](/src/makefile), run commmand
```
make parameter_1={value_1} parameter_2={value_2} ..
```

Example command has been provided in the run.sh file.

Reults will be formed in the `results` folder, and model will be saved in `models` folder.

## Sample Dataset
A sample Cora graph is given in the data folder. The structure of input graph is as follows:
```
<Graph_name> //input to the model
    <test_set> 
        <1>...<n>
    <val_set> 
        <1>...<n>
    <train_set> 
        <1>...<n>
// Each folder in train/val/test set should have a graph.txt and graph_stats.txt. The number of required cuts can be modifited in the graph_stats file.
```
Also, to run on any new graph, you need to add the graph.txt and node_embedding.pt file in raw_data folder. 

## Adding a new dataset 
1. Add the `graph.txt` and `node_embedding.pt` file in the `raw_data` folder. You can use the [a.py](./raw_data/a.py) file to download and modify the dataset.
2. Create an experiment in the `data` folder containing the train, test, and validation subdirectories. You will need to add the `graph.txt` and `graph_stats.txt` file there.
3. Our framework supports training on multiple graphs and inferring on different graphs as well. Configure your experiment folder accordingly to take advantage of this flexibility. 