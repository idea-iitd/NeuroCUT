import os
import torch
import networkx as nx
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.datasets import HeterophilousGraphDataset, Actor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Coauthor, LINKXDataset, HeterophilousGraphDataset


                

# Step 1: Load Citeseer dataset


dataset_name = input("Enter the dataset name: ") # prompt user to enter dataset name
if dataset_name == "cora_lc":
    dataset = Planetoid(root='../data_cora_lc', name='Cora', transform=T.NormalizeFeatures())
elif dataset_name == "citeseer_lc":
    dataset = Planetoid(root='../data_citeseer_lc', name='CiteSeer', transform=T.NormalizeFeatures())
elif dataset_name == "coauthor":
    dataset = Coauthor("../data_coauthor",name='Physics')

else:
    print("Dataset unavailable. Include the dataset from torch geometric if available; otherwise, download it.")


data = dataset[0]

# breakpoint()
# Step 2: Convert to NetworkX graph
graph = to_networkx(data, to_undirected=True)

# Remove self-loop edges
graph.remove_edges_from(nx.selfloop_edges(graph))

# Step 3: Find the largest connected component
largest_cc = max(nx.connected_components(graph), key=len)

# Step 4: Extract the subgraph corresponding to the largest connected component
largest_cc_graph = graph.subgraph(largest_cc)

print(largest_cc_graph)

nodes = list(largest_cc_graph.nodes())

# Step 5: Get node features for nodes in the largest connected component
node_features = data.x[nodes]

# Step 6: Save graph and node features
os.makedirs(dataset_name, exist_ok=True)
torch.save(node_features, f'{dataset_name}/node_embeddings.pt')

# Create a mapping between original node indices and indices in largest connected component
node_mapping = {node: idx for idx, node in enumerate(largest_cc_graph.nodes())}

# Save graph in .txt file
with open(f'{dataset_name}/graph.txt', 'w') as f:
    for edge in largest_cc_graph.edges():
        adjusted_edge = (node_mapping[edge[0]], node_mapping[edge[1]])
        f.write(f"{adjusted_edge[0]} {adjusted_edge[1]}\n")

print(f"{dataset_name} data saved successfully.")