
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


def load_data(data_dir, start_ts, end_ts):
	
    # Load the data from the csv files
	classes_csv = 'elliptic_txs_classes.csv'
	edgelist_csv = 'elliptic_txs_edgelist.csv'
	features_csv = 'elliptic_txs_features.csv'

    # Load the data from the csv files
	classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
	edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col = 'txId1') # directed edges between transactions
	features = pd.read_csv(os.path.join(data_dir, features_csv), header = None, index_col = 0) # features of the transactions
	
	print('\nData loaded from the csv files\n')

    # Extract information from the data
	num_features = features.shape[1] # number of features
	num_tx = features.shape[0] # number of transactions
	total_tx = list(classes.index) # list of all transactions
	
    print('Number of features:', num_features)

	# select only the transactions which are labelled
	labelled_classes = classes[classes['class'] != 'unknown']
	labelled_tx = list(labelled_classes.index)

	print('\nonly the transactions which are labelled selected')

	# to calculate a list of adjacency matrices for the different timesteps

	adj_mats = []
	features_labelled_ts = []
	classes_ts = []
	num_ts = 49 # number of timestamps from the paper

	# Convert total_tx to a mapping of transaction ID to matrix index
	tx_to_index = {tx_id: idx for idx, tx_id in enumerate(total_tx)}

	print('\nConvert total_tx to a mapping of transaction ID to matrix index')

	#For every timestep between start and end, it prepares an adj matrix

	for ts in range(start_ts, end_ts):
    	
		features_ts = features[features[1] == ts + 1]
            
		tx_ts = list(features_ts.index)
        
		labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]

        # Using lil_matrix for easier incremental construction
        
		adj_mat = lil_matrix((num_tx, num_tx), dtype=int)

        
		edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
        
		for i in range(edgelist_labelled_ts.shape[0]):
            
			tx1 = edgelist_labelled_ts.index[i]
            
			tx2 = edgelist_labelled_ts.iloc[i]['txId2']
            
			if tx1 in tx_to_index and tx2 in tx_to_index:  # Check if both tx exist in the mapping
                
				adj_mat[tx_to_index[tx1], tx_to_index[tx2]] = 1

		print('end of loop')

        # Convert back to csr_matrix for efficient arithmetic and slicing

		adj_mat_csr = adj_mat.tocsr()
        
		adj_mats.append(adj_mat_csr)

		print('adjacency matrix done')

        # Filter features and classes for labelled transactions of this timestep
        
		features_labelled_ts.append(features.loc[labelled_tx_ts])

		print('features labelled done')
        
		classes_ts.append(classes.loc[labelled_tx_ts])

		print('classes ts done')

	return adj_mats, features_labelled_ts, classes_ts