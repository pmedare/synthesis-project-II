import networkx as nx
import numpy as np

def laplacian_eigenmaps(G, dim=3):
    L = nx.laplacian_matrix(G).asfptype()
    eigvals, eigvecs = np.linalg.eigh(L.todense())  # Use eigh, as L is symmetric
    return eigvecs[:, 1:dim+1]  # skip the first eigenvector (eigenvalue 0)