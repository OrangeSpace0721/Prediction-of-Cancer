# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:27:10 2021

@author: xw3g19
"""
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.optimize import minimize_scalar
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv("breast-cancer-wisconsin.data", names=["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]).dropna()
data = data[~data["Bare Nuclei"].str.contains("\?")].astype(int)
independent = ["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

X = data[independent]
y = data.Class.replace(2, 0).replace(4, 1)
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.25, random_state = 0)
X_array = np.array(X_train)
y_array = np.array(y_train)
X_test_array = np.array(X_test)
y_test_array = np.array(y_test)


def p(w, b, x):
    p = 1/(1+np.exp(-w@x-b))
        
    return p

def obj(w, b, x, y, lambd):
    n = 0
    obj_sum = 0
    while n < len(x):
        log_loss = np.average(y[n]*np.log(p(w, b, x[n])+1e-10) + (1-y[n])*np.log(1-p(w, b, x[n])+1e-10))
        obj_sum = obj_sum + log_loss
        n=n+1
    
    obj_sum = (-1/(n))*obj_sum + lambd*np.linalg.norm(w)**2
    
    return obj_sum

def gradobj(w, b, x, y, lambd):
    n = 0
    m = 0
    gradW_sum = np.zeros(9)
    gradb_sum = 0
    
    while n < len(x):
        gradW_sum = gradW_sum + (p(w, b, x[n])-y[n])*x[n]
        n=n+1
    
    while m < len(x):
        gradb_sum = gradb_sum + p(w, b, x[m] - y[m])
        m=m+1
        
    gradW = (1/(n))*gradW_sum + 2*lambd*w
    gradb = (1/(m))*gradb_sum
    
    return gradW, gradb
    
def GD(w, b, iteration, alpha):
    i=1
    fx = 0
    fx_list = list(range(iteration+1))
    fx_list[0] = obj(w, b, X_array, y_array, 0.5)
    
    while i < iteration+1:
        w = w - alpha*gradobj(w, b, X_array, y_array, 0.5)[0]
        b = b - alpha*gradobj(w, b, X_array, y_array, 0.5)[1]
        fx = obj(w, b, X_array, y_array, 0.5)
        fx_list[i] = fx
        i = i+1
    
    return w, b, fx_list

w, b, fx = GD(np.zeros(9), 0, 100, 0.1)
a=[*range(0, 101, 1)]
plt.plot(a,fx,label='gradient descent')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()
j = 0
y_predict = list(range(len(y_test)))
p_list = list(range(len(y_test)))
rmse = 0
while j < len(y_test):
    p_model = p(w, b, X_test_array[j])
    p_list[j] = p_model
    if p_model > 0.5:
        y_predict[j] = 1
    if p_model < 0.5:
        y_predict[j] = 0
    rmse = rmse + (p_model - y_test_array[j])**2
    j=j+1

y_predict = np.array(y_predict)
rmse = (rmse/j)**(1/2)
print("predicted y is", y_predict)
print("actual y is", y_test_array)
acc_vector = y_predict - y_test_array
accuracy = (len(y_predict) - np.count_nonzero(acc_vector)) / len(y_predict) * 100
print("it has an absolute accuracy of", accuracy, "%")
auc = metrics.roc_auc_score(y_test, p_list)
print("the auc score is", auc)
print("the root mean square error is", rmse)

def GD_LS(w, b, iteration):
    i=1
    fx_list = list(range(iteration+1))
    fx_list[0] = obj(w, b, X_array, y_array, 0.5)
    
    while i < iteration+1:
        grad = gradobj(w, b, X_array, y_array, 0.5)
        res = minimize_scalar(lambda alpha: obj(w-alpha*grad[0], b-alpha*grad[1], X_array, y_array, 0.5))
        alpha = res.x
        w = w - alpha*grad[0]
        b = b - alpha*grad[1]
        fx = obj(w, b, X_array, y_array, 0.5)
        fx_list[i] = fx
        i = i+1
        
    return w, b, fx_list

w, b, fx = GD_LS(np.zeros(9), 0, 40)
a=[*range(0,41,1)]
plt.plot(a,fx,label='gradient descent with line search')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()
j = 0
y_predict = list(range(len(y_test)))
p_list = list(range(len(y_test)))
rmse = 0
while j < len(y_test):
    p_model = p(w, b, X_test_array[j])
    p_list[j] = p_model
    if p_model > 0.5:
        y_predict[j] = 1
    if p_model < 0.5:
        y_predict[j] = 0
    rmse = rmse + (p_model - y_test_array[j])**2
    j=j+1

y_predict = np.array(y_predict)
rmse = (rmse/j)**(1/2)
print("predicted y is", y_predict)
print("actual y is", y_test_array)
acc_vector = y_predict - y_test_array
accuracy = (len(y_predict) - np.count_nonzero(acc_vector)) / len(y_predict) * 100
print("it has an absolute accuracy of", accuracy, "%")
auc = metrics.roc_auc_score(y_test, p_list)
print("the auc score is", auc)
print("the root mean square error is", rmse)
    
def GD_HB(w, b, iteration, beta):
    i=1
    fx = 0
    fx_list = list(range(iteration+1))
    fx_list[0] = obj(w, b, X_array, y_array, 0.5)
    w_list = list(range(iteration+1))
    w_list[0] = w
    b_list = list(range(iteration+1))
    b_list[0] = b
    while i < iteration+1:
        grad = gradobj(w, b, X_array, y_array, 0.5)
        res = minimize_scalar(lambda alpha: obj(w-alpha*grad[0], b-alpha*grad[1], X_array, y_array, 0.5))
        alpha = res.x
        
        if i == 1:
            bw = 0
            bb = 0
        else:
            bw = beta*(w_list[i-1]-w_list[i-2])
            bb = beta*(b_list[i-1]-b_list[i-2])
            
        w = w - alpha*grad[0] + bw
        w_list[i] = w
        b = b - alpha*grad[1] + bb
        b_list[i] = b
        fx = obj(w, b, X_array, y_array, 0.5)
        fx_list[i] = fx
        i = i+1
        
    return w, b, fx_list

w, b, fx = GD_HB(np.zeros(9), 0, 40, 0.2)
a=[*range(0,41,1)]
plt.plot(a,fx,label='gradient descent with heavy ball')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()
j = 0
y_predict = list(range(len(y_test)))
p_list = list(range(len(y_test)))
rmse = 0
while j < len(y_test):
    p_model = p(w, b, X_test_array[j])
    p_list[j] = p_model
    if p_model > 0.5:
        y_predict[j] = 1
    if p_model < 0.5:
        y_predict[j] = 0
    rmse = rmse + (p_model - y_test_array[j])**2
    j=j+1

y_predict = np.array(y_predict)
rmse = (rmse/j)**(1/2)
print("predicted y is", y_predict)
print("actual y is", y_test_array)
acc_vector = y_predict - y_test_array
accuracy = (len(y_predict) - np.count_nonzero(acc_vector)) / len(y_predict) * 100
print("it has an absolute accuracy of", accuracy, "%")
auc = metrics.roc_auc_score(y_test, p_list)
print("the auc score is", auc)
print("the root mean square error is", rmse)

def GD_NV(w, b, iteration):
    i=1
    fx = 0
    fx_list = list(range(iteration+1))
    fx_list[0] = obj(w, b, X_array, y_array, 0.5)
    w_list = list(range(iteration+1))
    w_list[0] = w
    b_list = list(range(iteration+1))
    b_list[0] = b
    while i < iteration+1:
        grad = gradobj(w, b, X_array, y_array, 0.5)
        res = minimize_scalar(lambda alpha: obj(w-alpha*grad[0], b-alpha*grad[1], X_array, y_array, 0.5))
        alpha = res.x
        
        if i == 1:
            yw = w
            yb = b
        else:
            yw = w + ((i-1)/(i+2))*(w_list[i-1]-w_list[i-2])
            yb = b + ((i-1)/(i+2))*(b_list[i-1]-b_list[i-2])
            
        w = yw - alpha*gradobj(yw, b, X_array, y_array, 0.5)[0]
        w_list[i] = w
        b = yb - alpha*gradobj(w, yb, X_array, y_array, 0.5)[1]
        b_list[i] = b
        fx = obj(w, b, X_array, y_array, 0.5)
        fx_list[i] = fx
        i = i+1
        
    return w, b, fx_list

w, b, fx = GD_NV(np.zeros(9), 0, 100)
a=[*range(0,101,1)]
plt.plot(a,fx,label='Nesterov gradient descent')
plt.xlabel("iterations")
plt.ylabel("cost")
plt.legend()
plt.show()

j = 0
y_predict = list(range(len(y_test)))
p_list = list(range(len(y_test)))
rmse = 0
while j < len(y_test):
    p_model = p(w, b, X_test_array[j])
    p_list[j] = p_model
    if p_model > 0.5:
        y_predict[j] = 1
    if p_model < 0.5:
        y_predict[j] = 0
    rmse = rmse + (p_model - y_test_array[j])**2
    j=j+1

y_predict = np.array(y_predict)
rmse = (rmse/j)**(1/2)
print("predicted y is", y_predict)
print("actual y is", y_test_array)
acc_vector = y_predict - y_test_array
accuracy = (len(y_predict) - np.count_nonzero(acc_vector)) / len(y_predict) * 100
print("it has an absolute accuracy of", accuracy, "%")
auc = metrics.roc_auc_score(y_test, p_list)
print("the auc score is", auc)
print("the root mean square error is", rmse)
   
def hesobj(w, b, x):
    i = 0
    size = np.shape(x)[0]
    B = np.zeros((size,size))
    grad = gradobj(w,b,x,y_array,0.5)[1]
    psum = 0
    while i < size:
        p_now = p(w, b, x[i])
        B[i][i] = p_now*(1-p_now)
        psum = psum + p_now*(1 - p_now)
        i = i+1
    
    hesw = 1/size * np.transpose(x)@B@x
    hesb = size * grad/psum
    
    return hesw, hesb

def newtons(w, b, x, iteration):
    i=1
    fx = 0
    fx_list = list(range(iteration+1))
    fx_list[0] = obj(w, b, X_array, y_array, 0.5)
    
    while i < iteration+1:
        invw = np.linalg.inv(hesobj(w, b, x)[0])
        grad = gradobj(w, b, X_array, y_array, 0.5)
        dw = invw@grad[0]
        db = hesobj(w, b, x)[1]
        res = minimize_scalar(lambda alpha: obj(w-alpha*dw, b-alpha*db, X_array, y_array, 0.5))
        alpha = res.x
        w = w - alpha*dw
        b = b - alpha*db
        fx = obj(w, b, X_array, y_array, 0.5)
        fx_list[i] = fx
        i = i+1
        
    return w, b, fx_list

w, b, fx = newtons(np.zeros(9), 0, X_array, 30)
a=[*range(0,31,1)]
plt.plot(a,fx,label="Newton's method")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()
j = 0
y_predict = list(range(len(y_test)))
p_list = list(range(len(y_test)))
rmse = 0
while j < len(y_test):
    p_model = p(w, b, X_test_array[j])
    p_list[j] = p_model
    if p_model > 0.5:
        y_predict[j] = 1
    if p_model < 0.5:
        y_predict[j] = 0
    rmse = rmse + (p_model - y_test_array[j])**2
    j=j+1

y_predict = np.array(y_predict)
rmse = (rmse/j)**(1/2)
print("predicted y is", y_predict)
print("actual y is", y_test_array)
acc_vector = y_predict - y_test_array
accuracy = (len(y_predict) - np.count_nonzero(acc_vector)) / len(y_predict) * 100
print("it has an absolute accuracy of", accuracy, "%")
auc = metrics.roc_auc_score(y_test, p_list)
print("the auc score is", auc)
print("the root mean square error is", rmse)
