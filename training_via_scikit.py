
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from stringcolor import *
import numpy as np

iris=datasets.load_iris()


x=iris.data[:,[2,3]]
y=iris.target
#test_size=0.3 means 30 percent of test data and 70 percent train data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
# print(x)
# print(np.unique(y))
# print(x_train)
# print(x_test)
sc=StandardScaler()
sc.fit(x_train)
#Using the fit method, StandardScaler estimated the parameters
#µ (sample mean) and σ (standard deviation) for each feature dimension from the
#training data
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)
ppn=Perceptron(n_iter_no_change=30,eta0=0.1,random_state=0)
ppn.fit(x_train_std,y_train)
y_pred=ppn.predict(x_test_std)
print(y_pred)
print(y_test)
mismatch=zip(y_pred,y_train)
# mismatch= next(i for i, (el1,el2) in enumerate(zip(y_pred,y_train)) if el1 == el2)
# print(list(mismatch))
# print(mismatch)

e=filter(lambda x: x[0]!=x[1],mismatch)
print(list(e))
print(" mismatch samples %d "%(y_test!=y_pred).sum())
# print(x_train_std)
print(accuracy_score(y_test,y_pred)*100)
