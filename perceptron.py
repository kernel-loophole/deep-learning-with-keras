import numpy as np
from tqdm import tqdm
import pandas as pd

class Perceptron(object):
    def __init__(self,eta=0.01,n_iter=10):
        self.eta=eta
        self.n_iter=n_iter
    def fit(self,X,Y):
        self.w_=np.zeros(1+X.shape[1])
        # print(self.w_)
        self.error=[]
        #calculate the error in each epochs
        # n_iter is number of epochs 
        for _ in tqdm(range(self.n_iter),desc='Training',unit='epoch',position=0):
            errors=0
            # print(list(zip(X,Y)))
            for xi,target in zip(X,Y):
                # print(xi,target)
                update=self.eta*(target - self.predict(xi))
                # print(update)
                self.w_[1:]+=update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.error.append(errors)
        print(self.error)
        return self
    
    def net_input(self, X):
    
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):

        return np.where(self.net_input(X) >= 0.0, 1, -1)        
if __name__=="__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # print(df)

    y = df.iloc[0:100, 4].values

    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    # print(y)
    # print(X)
    ppn = Perceptron(eta=0.1, n_iter=5)
    ppn.fit(X, y)
