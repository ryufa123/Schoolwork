# -*- coding: utf-8 -*-
"""
Created on Sun May 30 08:48:19 2021

@author: LIKAI
"""
import csv
import random
import operator
import numpy as np

def distance(d1, d2):
    res = 0
    feature = ("Petal length","Petal width","Sepal length","Sepal width")#read feature
    for key in (feature): #derive distance
        res += (float(d1[key])-float(d2[key]))**2
    return res**0.5
  
#KNN    
K = 5  
def knn(data):
    ##1 derive distance
    ##2 Sort-Ascending
    ##3 Take the top K
    ##4 weighted average
    ##1 derive distance
    res = [
            {"result":train['class'], "distance": distance(data, train)}
            for train in train_set
          ]
    ##Use derivation again, assign the flower type to result,and the distance to distance 
    ##2 In ascending order based on distance, the closest one is in the front
    res = sorted(res, key = lambda item:item['distance'])
    #print(res) 
    
    ##3
    res2 = res[0:K]
    #print(res2)
    
    ##4
    result = {'Iris-setosa': 0, 'Iris-versicolor':0, 'Iris-virginica': 0}#(Initial weight)
    
    sum = 0
    for r in res2:                 
        sum+= r['distance'] 
    for r in res2:                
        result[r['result']] += 1 - r['distance']/sum   
    resultCount1= sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    ##Sort according to the size of the weighted result, reverse = True for descending order              
    return (resultCount1[0][0])
    ##Return the maximum value (the two data in resultcount1 are the category and distance values of Iris)

##test
if __name__ == "__main__":
    a = []
    for i in range(20):
        with open('iris.csv','r') as file:  
            reader = csv.DictReader(file)   ##read file as a dictionary
                
                #for row in reader:
                    #print(row)    
            datas = [row for row in reader] ##datas is a big list
                #print(datas)
            ##grouping
            random.shuffle(datas)  ##shuffle the dataset 
            n = len(datas)//2       ##divide the dataset into two part
            test_set = datas[0:n]   
            train_set = datas[n:]
        correct = 0                       
        for test in test_set:       
            result = test['class']
            result2 = knn(test)
            if result  == result2:
                correct+=1
        print("Correct rate：%.3f"%(correct/len(test_set)))
        print("==================")
        a.append(float((correct/len(test_set))))
        
        
        i = i+1
    #print(a)
    print("Mean is :{:.3}".format(np.mean(a))) 
    print("==============================")
    print("Variance is :{:.3}".format(np.var(a)))
            
                
    
    
                
    