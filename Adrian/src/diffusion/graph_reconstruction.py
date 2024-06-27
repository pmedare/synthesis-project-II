import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

def reconstruct_graph(embeddings, threshold=0.5):
    sim_matrix = cosine_similarity(embeddings)
    adj_matrix = (sim_matrix > threshold).astype(int)
    G = nx.from_numpy_matrix(adj_matrix)
    return G