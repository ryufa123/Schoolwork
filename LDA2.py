# -*- coding: utf-8 -*-
"""
Created on Sun May 30 19:16:54 2021

@author: LIKAI
"""

import numpy as np
import pandas as pd
import random


class LinearDiscrtAnays(object):
    def __init__(self):
        self.Xi_means = 0               ##Mean vector of each category
        self.Xbar = 0                   ##The overall mean vector
        self.covMatrix = []             ##Covariance matrix for each category
        self.covariance_ = 0            ##Overall covariance matrix
        self.X = 0                      ##Training data
        self.y = 0                      ##Classification label of training data
        self.classes_ = 0               ##Specific category
        self.priors_ = 0                ##Prior probability of each category
        self.n_samples = 0              ##Number of samples of training data
        self.n_features = 0             ##Features of training data
        self.n_components = 0           ##Features
        self.w = 0                      ##Feature vector
    
    #init feature   
    """calculate params, including:
        0. X, y
        1. n_samples, n_features;
        2. classer, priors_;
        3. Xi_means, Xbar, covMatrix;
    """
    def _params_init(self, X, y):
        ##0、Assign X and y
        self.X, self.y = X, y
        ##1、Calculate the number of samples and the number of features
        self.n_samples, self.n_features = X.shape
        ##2、Calculate the category value and the prior probability of each category
        self.classes_, yidx = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y) / self.n_samples
        ##3、Calculate the mean of each category
        means = np.zeros((len(self.classes_), self.n_features))
        np.add.at(means, yidx, X)
        self.Xi_means = means / np.expand_dims(np.bincount(y), 1)
        ##4、Calculate the covariance matrix of each category and the overall covariance matrix
        self.covMatrix = [np.cov(X[y == group].T) \
                          for idx, group in enumerate(self.classes_)]
        self.covariance_ = sum(self.covMatrix) / len(self.covMatrix)
        ##5、Calculate the population mean vector
        self.Xbar = np.dot(np.expand_dims(self.priors_, axis=0), self.Xi_means)
        return 
    
    ##train
    def train(self, X, y, n_components=None):
        ##init
        self._params_init(X, y)
        ##Find the average divergence within a class
        Sw = self.covariance_
        ##Find the average divergence between classes
        Sb = sum([sum(y == group)*np.dot((self.Xi_means[idx,None] - self.Xbar).T, (self.Xi_means[idx,None] - self.Xbar)) \
                  for idx, group in enumerate(self.classes_)]) / (self.n_samples - 1)
        ##SVD finds the inverse matrix of Sw
        U,S,V = np.linalg.svd(Sw)
        Sn = np.linalg.inv(np.diag(S))
        Swn = np.dot(np.dot(V.T,Sn),U.T)
        SwnSb = np.dot(Swn,Sb)
        ##Find eigenvalues and eigenvectors, and take the real part
        la,vectors = np.linalg.eig(SwnSb)
        la = np.real(la)
        vectors = np.real(vectors)
        laIdx = np.argsort(-la)
        if n_components == None:
            n_components = len(self.classes_)-1
        ##Select eigenvalues and vectors
        lambda_index = laIdx[:n_components]
        w = vectors[:,lambda_index]
        self.w = w
        self.n_components = n_components
        return
    
    ##Find the matrix after projection
    def transform(self, X):
        return np.dot(X, self.w)
    
    ##Predict the classification situation, the probability of classification
    def predict_prob(self, X):
        ##Seeking the inverse of the overall covariance
        Sigma = self.covariance_
        U,S,V = np.linalg.svd(Sigma)
        Sn = np.linalg.inv(np.diag(S))
        Sigman = np.dot(np.dot(V.T,Sn),U.T)
        ##Linear discriminant function value
        value = np.log(np.expand_dims(self.priors_, axis=0)) - \
        0.5*np.multiply(np.dot(self.Xi_means, Sigman).T, self.Xi_means.T).sum(axis=0).reshape(1,-1) + \
        np.dot(np.dot(X, Sigman), self.Xi_means.T)
        return value/np.expand_dims(value.sum(axis=1),1)
    
    ##Predict the classification situation and get the specific classification value
    def predict(self, X):
        pValue = self.predict_prob(X)
        return np.argmax(pValue, axis=1)
    
def read_csv(csvPath, train_number):
    trainFeature = []
    trainlabel = []
    valFeature = []
    vallabel = []
    ##read csv
    data_df = pd.read_csv(csvPath)
    feature = data_df[['Petal length', 'Petal width', 'Sepal length', 'Sepal width']].values
    labelName = data_df['class'].values
    idxList = list(range(len(data_df)))
    random.shuffle(idxList)##shuffle index list
    #print(idxList)
    trainIdxList = idxList[:train_number]##The first half is used for training 
    valIdxList = idxList[train_number:]##and the second half is used for testing
    ##Loop to add each tag class to the array
    for trainIdx in trainIdxList:
        trainFeature.append(feature[trainIdx,:])
        trainlabel.append(classNameDict[labelName[trainIdx]])
        #print(trainFeature)
        #print(trainIdxList)
        #print(trainlabel)
    for valIdx in valIdxList:
        valFeature.append(feature[valIdx,:])
        vallabel.append(classNameDict[labelName[valIdx]])
        #print(valFeature)
    return np.array(trainFeature), np.array(trainlabel), np.array(valFeature), np.array(vallabel)
    

if __name__ == "__main__":
    ##0、prepare Iris dataset
    '''
    ##Ignore here!!!
    (same way as knn, but not used, hard to handle dictionary)
    with open('iris.csv','r') as file:  
        reader = csv.DictReader(file)   
    #for row in reader:
        #print(row)    
        datas = [row for row in reader] 
        print(datas)
    random.shuffle(datas)   
    n = len(datas)//2  
    test_set = datas[0:n]   ##testset
    train_set = datas[n:]   ##trainset
    '''
    a = []
    for i in range(10):
        
        classNameDict = {'Iris-setosa': 1 , 'Iris-versicolor': 0 ,'Iris-virginica': 2}
        print("==============================")
        ##The first 75 are used for training and the last 75 are used for testing
        trainFeature, trainlabel, valFeature, vallabel = read_csv(r'C:\Users\Mechrevo\Desktop\春季课程各种信息\適応的メディア処理\iris.csv',75)
        lda = LinearDiscrtAnays()
        lda.train(trainFeature,trainlabel)
        pre = lda.predict(valFeature)
        a.append(sum(pre==vallabel)/valFeature.shape[0])
        print("Correct rate：{:.3f}".format(sum(pre==vallabel)/valFeature.shape[0]))
        i=i+1
    print("Mean is :{:.3}".format(np.mean(a))) 
    print("==============================")
    print("Variance is :{:.3}".format(np.var(a)))
    #print("datatype",type(a))            
          
          