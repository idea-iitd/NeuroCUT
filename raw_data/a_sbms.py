import os
import torch
import networkx as nx
import json
import matplotlib.pyplot as plt

# from torch_geometric.datasets import HeterophilousGraphDataset, Actor


# Step 1: Load Citeseer dataset
# dataset = Planetoid(root='data/Citeseer', name='Citeseer', transform=T.NormalizeFeatures())

# dataset = Actor(root='data/actor')
directory = 'SBM_500_vary'

for filename in os.listdir(directory):
    f1 = os.path.join(directory, filename)

    num_nodes=500
    tgraph=nx.read_edgelist(f1+'/graph.txt', nodetype=int)
    graph=nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(tgraph.edges())

    # Remove self-loop edges
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Step 3: Find the largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)

    # Step 4: Extract the subgraph corresponding to the largest connected component
    largest_cc_graph = graph.subgraph(largest_cc)

    print(largest_cc_graph)

    nodes = list(largest_cc_graph.nodes())


    # Create a mapping between original node indices and indices in largest connected component
    node_mapping = {node: idx for idx, node in enumerate(largest_cc_graph.nodes())}

    # Save graph in .txt file
    with open(f1+'/graph.txt', 'w') as f:
        for edge in largest_cc_graph.edges():
            adjusted_edge = (node_mapping[edge[0]], node_mapping[edge[1]])
            f.write(f"{adjusted_edge[0]} {adjusted_edge[1]}\n")

    # Save stats in .txt file
    with open(f1+'/graph_stats.txt', 'w') as f:
        x = {"num_nodes": largest_cc_graph.number_of_nodes(), "num_edges":largest_cc_graph.number_of_edges() , "num_cuts": 5}
        json.dump(x,f)

    # Print the number of nodes and edges in the generated graph
    print("Number of nodes:", len(largest_cc_graph.nodes))
    print("Number of edges:", len(largest_cc_graph.edges))

    # plot the graph
    nx.draw(largest_cc_graph, with_labels=False)
    # save the graph image
    plt.savefig(f1+"/graph.png")
    plt.clf()
    # save the stats about graph in .txt file
    with open(f1+'/stats_updated.txt','w') as f:
        # total number of nodes
        f.write(f'total_nodes: {len(largest_cc_graph.nodes)}')
        f.write('\n')
        # total number of edges
        f.write(f'total_edges: {len(largest_cc_graph.edges)}')
        f.write('\n')
        # total connected components
        total_connected_components = nx.number_connected_components(largest_cc_graph)
        f.write(f'total_connected_components: {total_connected_components}')
        f.write('\n')
        # avg degree
        avg_degree = 2*len(largest_cc_graph.edges)/len(largest_cc_graph.nodes)
        f.write(f'avg_degree: {avg_degree}')
        f.write('\n')
        # highest degree, lowest degree
        highest_degree = max(dict(largest_cc_graph.degree).values())
        lowest_degree = min(dict(largest_cc_graph.degree).values())
        f.write(f'highest_degree: {highest_degree}')
        f.write('\n')
        f.write(f'lowest_degree: {lowest_degree}')
        f.write('\n')
        # avg clustering coefficient
        avg_clustering_coeff = nx.average_clustering(largest_cc_graph)
        f.write(f'avg_clustering_coeff: {avg_clustering_coeff}')
        f.write('\n')
        # avg degree centrality
        avg_degree_centrality = sum(nx.degree_centrality(largest_cc_graph).values())/len(nx.degree_centrality(largest_cc_graph))
        f.write(f'avg_degree_centrality: {avg_degree_centrality}')
        f.write('\n')
        # avg closeness centrality
        avg_closeness_centrality = sum(nx.closeness_centrality(largest_cc_graph).values())/len(nx.closeness_centrality(largest_cc_graph))
        f.write(f'avg_closeness_centrality: {avg_closeness_centrality}')
        f.write('\n')
        # avg betweenness centrality
        avg_betweenness_centrality = sum(nx.betweenness_centrality(largest_cc_graph).values())/len(nx.betweenness_centrality(largest_cc_graph))
        f.write(f'avg_betweenness_centrality: {avg_betweenness_centrality}')
        f.write('\n')

    print("SBM data saved successfully.")