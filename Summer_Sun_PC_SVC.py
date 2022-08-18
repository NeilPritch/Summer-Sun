# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 07:47:54 2022

@author: cnpritch
"""

import numpy as np
import gudhi
import persistencecurves as pc
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


#%%%%%%%
def calculate_diagram_gudhi(image):#Calculates the oth and 1st dimensional  persistence diagrams from an image
    Complex = gudhi.CubicalComplex(dimensions=(256,256), top_dimensional_cells=image) 
    Complex.persistence()
    Dgm0=Complex.persistence_intervals_in_dimension(0) # oth
    Dgm1=Complex.persistence_intervals_in_dimension(1) # 1st
    return Dgm0,Dgm1


data_train = np.load(r"myfile")['data']

data_test = np.load(r"myfile")['data']

#data_train and data_test are obtaines from catalogue of sun image data loacted at 
#https://github.com/bionictoucan/Slic/releases/tag/1.1.1, from the paper Fast Solar Image Classification Using Deep Learning and its Importance
#for Automation in Solar Physics by Armstron and Fletcher

#%%%

train_target_vector = data_train[:,0]

train_feature_vector = np.delete(data_train, 0,1)   

test_target_vector = data_test[:,0]

test_feature_vector = np.delete(data_test,0,1)   

#%%%%%%%%%%%%%%%%%
Keys = np.linspace(0,len(data_train), num = len(data_train), endpoint=False)
Train_Dict_PDs = {}
for i in range(len(data_train)):
        im  = train_feature_vector[i]
        Dgm0, Dgm1 = calculate_diagram_gudhi(im)
        im = 255-im
        IDgm0,IDgm1 = calculate_diagram_gudhi(im)
        Diagrams = [Dgm0,Dgm1,IDgm0,IDgm1]
        Train_Dict_PDs[Keys[i]] = Diagrams
        
        
Keys = np.linspace(0,len(data_test), num = len(data_test), endpoint=False)
Test_Dict_PDs = {}
for i in range(len(data_test)):
        im  = test_feature_vector[i]
        Dgm0, Dgm1 = calculate_diagram_gudhi(im)
        im = 255-im
        IDgm0,IDgm1 = calculate_diagram_gudhi(im)
        Diagrams = [Dgm0,Dgm1,IDgm0,IDgm1]
        Test_Dict_PDs[Keys[i]] = Diagrams
#%%%%%%%%%%
xtrain = np.zeros((11857,1024))

xtest = np.zeros((1318,1024))

for i in range(len(Train_Dict_PDs)): # computes the Persistence lifecurve for diagrams in training set.
    x = np.zeros(1024)
    Dgm0 = Train_Dict_PDs[i][0]
    Dgm0 = np.delete(Dgm0,-1,0)
    Dgm1 = Train_Dict_PDs[i][1]
    IDgm0 = Train_Dict_PDs[i][2]  
    IDgm0 = np.delete(IDgm0,-1,0) 
    IDgm1 = Train_Dict_PDs[i][3]
    D0 = pc.Diagram(Dgm =Dgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    D1 = pc.Diagram(Dgm =Dgm1, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    ID0 = pc.Diagram(Dgm =IDgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    ID1 = pc.Diagram(Dgm =IDgm1, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    dgm0 = D0.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    dgm1 = D1.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    Idgm0 = ID0.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    Idgm1 = ID1.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    dgm = np.concatenate((dgm0,dgm1,Idgm0,Idgm1), axis=0)
    xtrain[i,:] = dgm

for i in range(len(Test_Dict_PDs)): # computes persistence lifecurve for diagrams in testing set.
    x = np.zeros(1024)
    Dgm0 = Test_Dict_PDs[i][0]
    Dgm0 = np.delete(Dgm0,-1,0)
    Dgm1 = Test_Dict_PDs[i][1]
    IDgm0 = Test_Dict_PDs[i][2]
    IDgm0 = np.delete(IDgm0,-1,0)    
    IDgm1 = Test_Dict_PDs[i][3]
    D0 = pc.Diagram(Dgm =Dgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    D1 = pc.Diagram(Dgm =Dgm1, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    ID0 = pc.Diagram(Dgm =IDgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    ID1 = pc.Diagram(Dgm =IDgm1, globalmaxdeath = None, infinitedeath=None, inf_policy="remove")
    dgm0 = D0.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    dgm1 = D1.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    Idgm0 = ID0.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    Idgm1 = ID1.lifecurve(meshstart=0,meshstop=256,num_in_mesh=256)
    dgm = np.concatenate((dgm0,dgm1,Idgm0,Idgm1), axis=0)
    xtest[i,:] = dgm



#%%%%%%%%%
S = SVC()

S.fit(xtrain,train_target_vector)

pred = S.predict(xtest)

print(confusion_matrix(test_target_vector, pred))

















