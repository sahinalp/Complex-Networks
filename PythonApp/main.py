from PyQt5.QtWidgets import*
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QImage, QKeySequence,QPixmap,QPainter
from PyQt5 import QtTest
from PyQt5.QtCore import QObject

from PyQt5.Qt import Qt

import easygui
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

from helpers.functions import laplacian,normalize_eigenvectors,find_communities1
from helpers.kmeans import K_means

kmeans=K_means()

#%% Ui

class loadui(QMainWindow):
    
    
    def __init__(self):
        super().__init__()
        loadUi("main.ui",self)

        self.G=nx.Graph()
        self.data=None
        self.Q_dict={}
        self.labels_dict={}
        self.labels=None
        
        self.pushButton_loadData_json.clicked.connect(self.loadDataJson)
        self.pushButton_create_graph.clicked.connect(self.createGraph)
        self.pushButton_find_optimum.clicked.connect(self.findOptimum)
        self.pushButton_community.clicked.connect(self.findCommunity)

        self.pushButton_optimum_save.clicked.connect(self.saveOptimum)
        self.pushButton_community_save.clicked.connect(self.saveCommunity)
        
    def loadDataJson(self):
        try:
            self.path = easygui.fileopenbox(msg="Please locate the json file",default='*.json',filetypes='*.json')
            with open(self.path,"r",encoding="utf-8") as f:
                self.data=json.load(f)
            self.pushButton_create_graph.setEnabled(True)
            self.spinBox_upper.setEnabled(True)
            self.pushButton_loadData_json.setEnabled(False)
            self.pushButton_loadData_txt.setEnabled(False)
            self.spinBox_upper.setMaximum(len(self.data))
            self.spinBox_k.setMaximum(len(self.data)-1)
            self.spinBox_upper.setValue(len(self.data))
        except:
            pass
        
    
    def createGraph(self):
        count=0
        limit=self.spinBox_upper.value()+1

        for i in self.data:
            count+=1
            if count==limit:
                break
            lengthOfCast=len(self.data[i]['CastId'])
            for j in range(0,lengthOfCast-1):
                val1=self.data[i]['CastId'][j]
                for k in range(j+1,lengthOfCast):
                    val2=self.data[i]['CastId'][k]
                    self.G.add_edge(int(val1),int(val2))
        
        
        plt.figure(figsize=(20,20))
        nx.draw(self.G,with_labels=False,width=0.5)
        plt.savefig("figures/graph.png")
        self.graph_base.setPixmap(QtGui.QPixmap("figures/graph.png"))
        
        self.pushButton_find_optimum.setEnabled(True)
        self.pushButton_community.setEnabled(True)
        self.spinBox_k.setEnabled(True)
        


    def findOptimum(self):
        L = laplacian(self.G)
        self.Q_dict,self.labels_dict=kmeans.k_means_optimum(self.G,L)

        plt.figure(figsize=(20,20))
        ax=sns.lineplot(data=self.Q_dict)
        ax.set(xlabel='Cluster Count',ylabel='Modularity')
        ax.axes.invert_xaxis()
        plt.savefig("figures/optimum.png")
        self.graph_optimum.setPixmap(QtGui.QPixmap("figures/optimum.png"))
        

        self.pushButton_optimum_save.setEnabled(True)
    
    def saveOptimum(self):
        
        df_Q=pd.DataFrame({'Cluster count':self.Q_dict.keys(),'Q':self.Q_dict.values()})
        df_Q.to_csv("Q values.csv")

        df_labels=pd.DataFrame({'Cluster count':self.labels_dict.keys(),'Labels':self.labels_dict.values()})
        df_labels.to_csv("Labels.csv")
    
    def findCommunity(self):
        k=self.spinBox_k.value()
        L = laplacian(self.G)
        # _, eig_vectors = np.linalg.eig(L)
        _, eig_vectors = sp.linalg.eigs(L, k)
        X = eig_vectors.real
        X = np.apply_along_axis(normalize_eigenvectors, 0, X)
        self.labels=kmeans.k_means(X,k)
        # self.labels=find_communities1(self.G,labels)

        plt.figure(figsize=(20,20))
        nx.draw(self.G,node_color=self.labels,with_labels=False,width=0.5)
        plt.savefig("figures/kmeans_community.png")
        self.graph_community.setPixmap(QtGui.QPixmap("figures/kmeans_community.png"))
        

        self.pushButton_community_save.setEnabled(True)


    def saveCommunity(self):
        df_labels=pd.DataFrame({'nodes':self.G.nodes(),'community':self.labels})
        df_labels.to_csv("LabelsOfCommunity.csv")


app = QApplication([])
window = loadui()
window.show()
app.exec_()
# %%
