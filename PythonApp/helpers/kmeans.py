from sklearn.cluster import KMeans
from helpers.functions import find_communities1,modularity
import scipy.sparse as sp
import numpy as np
from helpers.functions import laplacian,normalize_eigenvectors,find_communities1

class K_means:
    def __init__(self,countdown=10):
        self.ct=countdown
        
    def k_means_optimum(self,G,L):
        Q_dict={}
        labels_dict={}
        countdown=self.ct
        count=1
        length_G=len(G.nodes)

        while True:
        
            _, eig_vectors = sp.linalg.eigs(L, length_G-count)
            X = eig_vectors.real
            X = np.apply_along_axis(normalize_eigenvectors, 0, X)
            kmeans = KMeans(n_clusters=length_G-count, random_state=0, n_init="auto").fit(X)

            labels = kmeans.labels_

            comm=find_communities1(G,labels)
            Q=modularity(G,comm,length_G-count)

            if (count!=1 and Q_dict[list(Q_dict.keys())[-1]]>Q):
                countdown-=1
            else:
                countdown=self.ct

            Q_dict[length_G-count]=Q
            labels_dict[length_G-count]=labels
            if length_G-count<500:
                count+=1
            else:
                count*=2
            if count>=length_G or countdown==0:
                break
        
        return Q_dict,labels_dict
    
    def k_means(self,X,k):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        labels = kmeans.labels_
        return labels
