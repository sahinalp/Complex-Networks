import numpy as np
import networkx as nx
import copy


def laplacian(G):
    

    D = np.diag(np.sum(np.array(nx.adjacency_matrix(G).todense()), axis=1))
    W = nx.adjacency_matrix(G).toarray()

    assert D.shape == W.shape, "Shapes of D and W don't match."

    L = None

    I = np.ones(D.shape)
    D_inv_root = np.linalg.inv(np.sqrt(D))

    L = I - np.dot(D_inv_root, W).dot(D_inv_root)
    

    return L

def normalize_eigenvectors(e):
    
    return e/np.sqrt(np.sum(e**2))

def find_communities(G):
    communities = {}
    temp_count=0
    n=len(G.nodes())
    for idx1,node1 in enumerate(G.nodes()):
        if n==len(communities):
            break
        count=temp_count+1
        for idx2,node2 in enumerate(G.nodes()):
            if node2 in communities:
                continue
            elif idx2==idx1:
                communities.update({node1:count})
            elif idx2<idx1:
                continue
            else:
                if (node1,node2) in G.edges():
                    communities.update({node2:count})
                    temp_count=count
    return communities,count

def find_communities1(G,labels):
    communities = {}
    count=0
    for node in G.nodes():
        communities[node] = labels[count]
        count+=1
    return communities

def group_communities(communities):
    count=0
    node_cmap=[]
    values={}
    for i in communities:
        temp=communities[i]
        if temp in values:
            node_cmap.append(values[temp])
        else:
            values.update({temp:count})
            node_cmap.append(values[temp])
            count+=1
            if count==1:
              count+=1
    print(count)
    return node_cmap

def group_communities2(communities):
    communities_dict = {}
    count=0
    for comunity in communities:
      count+=1
      for node in comunity:
        communities_dict.update({node:count})
    node_cmap=group_communities(communities_dict)
    print(count)
    return node_cmap

def calculate_modularity(M,n):
    Q=0
    for i in range(n):
        a=0
        for j in range(n):
            if i==j:
                e=M[i][j]
            else:
                a+=M[i][j]
        Q+=(e-(a*a))
    return Q

def modularity(G,communities,n):
    total_edge=G.edges.__len__()
    C=[[0]*n for i in range(n)]
    for edge in G.edges():
        c1=communities[edge[0]]
        c2=communities[edge[1]]
        if c2==c1:
            C[c1-1][c2-1]+=1
        else:
            C[c1-1][c2-1]+=1
            C[c2-1][c1-1]+=1

    M=copy.deepcopy(C)
    for i in range(n):
        for j in range(n):
            M[i][j]=M[i][j]/total_edge

    Q=calculate_modularity(M,n)
    return Q

def modularity_based(G,communities,n):
  total_edge=G.edges.__len__()
  C=[[0]*n for i in range(n)]
  for edge in G.edges():
      c1=communities[edge[0]]
      c2=communities[edge[1]]
      if c2==c1:
          C[c1-1][c2-1]+=1
      else:
          C[c1-1][c2-1]+=1
          C[c2-1][c1-1]+=1

  M=copy.deepcopy(C)
  for i in range(n):
      for j in range(n):
          M[i][j]=M[i][j]/total_edge
  
  Q=calculate_modularity(M,n)
  
  old_n=n
  while True:
      n=n-1
      if n==0:
        n=n+1
        break
      Q_old=Q
      new_C=[[0]*(n) for i in range(n)]
      for i in range(n):
          for j in range(n):
              new_C[i][j]=C[i+1][j+1]
              if (i,j)==(0,0):
                  new_C[i][j]+=C[i][j]
                  new_C[i][j]+=C[i][j+1]
              elif i==0:
                  new_C[i][j]+=C[i][j+1]
              elif j==0:
                  new_C[i][j]+=C[i+1][j]
      
      old_communities=copy.deepcopy(communities)
      for i in communities:
          if communities[i]==1:
              continue
          else:
              communities[i]-=1
      
      old_C=copy.deepcopy(C)
      C=copy.deepcopy(new_C)
      M=copy.deepcopy(C)
      for i in range(n):
          for j in range(n):
              M[i][j]=M[i][j]/total_edge
      
      Q=calculate_modularity(M,n)
      
      if Q_old>=Q:
          C=copy.deepcopy(old_C)
          communities=copy.deepcopy(old_communities)
          n=n+1
          break
  return communities,n