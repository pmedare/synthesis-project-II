
# Import necessary libraries
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.sparse import lil_matrix # If the graph is sparse (i.e., there are not many edges), a sparse representation can save a lot of memory.

def create_adjacency_matrix(edgelist, total_tx):
    # Create a mapping from transaction ID to index
    tx_to_index = {tx_id: index for index, tx_id in enumerate(total_tx)}
    
    num_tx = len(total_tx)
    adj_matrix = lil_matrix((num_tx, num_tx), dtype=int)
    
    for tx_id1, row in edgelist.iterrows():
        tx_id2 = row['txId2']
        
        # Map the transaction IDs to indices
        index1 = tx_to_index[tx_id1]
        index2 = tx_to_index[tx_id2]
        
        adj_matrix[index1, index2] = 1
        adj_matrix[index2, index1] = 1
    
    return adj_matrix

def create_networkx_graph(features, edgelist):
    """ Creates a NetworkX graph with nodes and edges """
    G = nx.DiGraph()
    
    # Add nodes along with their features
    for node_id, feature_values in features.iterrows():
        feature_dict = {f'feature_{i+1}': val for i, val in enumerate(feature_values)}
        G.add_node(node_id, **feature_dict)
    
    # Add edges
    for tx_id1, row in edgelist.iterrows():
        for tx_id2 in row:
            G.add_edge(tx_id1, tx_id2)
            
    return G

# Function to load the data from the csv files
def load_data(data_dir):
	
    # Load the data from the csv files
	classes_csv = 'elliptic_txs_classes.csv'
	edgelist_csv = 'elliptic_txs_edgelist.csv'
	features_csv = 'elliptic_txs_features.csv'

    # Load the data from the csv files
	classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
	edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col = 'txId1') # directed edges between transactions
	features = pd.read_csv(os.path.join(data_dir, features_csv), header = None, index_col = 0) # features of the transactions
	
	print('\nData loaded from the csv files\n')

	# Print all unique classes
	print('Unique classes:', classes['class'].unique())
	print()
	
	# Display first few rows of the classes dataframe
	print('Classes dataframe:')
	print(classes.head())
	print()

	# Display first few rows of the edgelist dataframe
	print('Edgelist dataframe:')
	print(edgelist.head())
	print()

	# Display first few rows of the features dataframe
	print('Features dataframe:')
	print(features.head())
	print()

    # Extract information from the data
	num_features = features.shape[1] # number of features
	num_tx = features.shape[0] # number of transactions
	total_tx = list(classes.index) # list of all transactions
	
	print('Number of features:', num_features)
	print('Number of transactions:', num_tx)
	print('First 10 transactions:', total_tx[:10])
	print()	

	labelled_classes = classes[classes['class'] != 'unknown'] # labels of the transactions which are known
	unlabelled_classes = classes[classes['class'] == 'unknown'] # labels of the transactions which are unknown
	
	labelled_tx = list(labelled_classes.index) # list of transactions which are labelled
	unlabelled_tx = list(unlabelled_classes.index) # list of transactions which are unlabelled

	print('Number of labelled transactions:', len(labelled_tx))
	print('Number of unlabelled transactions:', len(unlabelled_tx))
	print()
     
	# Create the adjacency matrix
	adj_matrix = create_adjacency_matrix(edgelist, total_tx)
	print("Adjacency Matrix Created!\n")
     
	# Print the shape of the adjacency matrix
	print('Shape of the adjacency matrix:', adj_matrix.shape)
     
	# Print the number of non-zero elements in the adjacency matrix
	print('Number of non-zero elements in the adjacency matrix:', adj_matrix.nnz)
	print()
    
    # Create the graph
	G = create_networkx_graph(features, edgelist)
	print("NetworkX Graph Created!\n")
     
	# Print the number of nodes in the graph
	print('Number of nodes in the graph:', G.number_of_nodes())
	print("Number of edges in the graph:", G.number_of_edges())
	print()
     
	# Print the attributes of the first node and the attributes of the first edge
	first_node = list(G.nodes)[0]
	# print('Attributes of the first node:', G.nodes[first_node])
	first_edge = list(G.edges)[0]
	# print('Attributes of the first edge:', G.edges[first_edge])
	# print()

	return adj_matrix, G

if __name__ == '__main__':
	
	data_dir = "/home/adriangar8/Documents/academia/year3/semester2/synthesis_project_II/data/elliptic_bitcoin_dataset"

	# adj_mats, features_labelled_ts, classes_ts = load_data(data_dir, start_ts, end_ts)
	load_data(data_dir)

	print('Data loaded successfully!\n')