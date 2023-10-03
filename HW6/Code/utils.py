import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filepath):
    df = pd.read_csv(filepath)
    X, y = df.iloc[:, :-1].values, df.iloc[:,-1].values
    return X, y

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def accuracy_score(y_true, y_pred):
    return np.mean(y_pred == y_true)

class LogisticRegression:
    def __init__(self, alpha = 0.1, epoch = 10000, delta = 0.0001, threshold = 0.5):
        self.alpha = alpha
        self.epoch = epoch
        self.delta = delta
        self.epoch = epoch
        self.threshold = threshold
        self.loss = []
       
    def _initialize(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        # weights = np.random.randn(X.shape[1], 1)
        weights = np.zeros(X.shape[1]).reshape((X.shape[1]),1)
        return weights, X
    
    def fit(self, X, y):
        self.w, self.X = self._initialize(X)
        r,c = self.X.shape
        self.y = y
        count = 0
        for i in range(self.epoch):
            z = sigmoid(np.dot(self.X, self.w))
            grad = 1/r * np.dot(self.X.T, z - np.reshape(self.y,(len(self.y),1)))
            old_w = self.w.copy()
            self.w = self.w - (self.alpha * grad)
            if np.linalg.norm(self.w - old_w) < self.delta:
                return self.w
            # print(i)
    def predict(self,X = None):
        if X is not None:
            X = np.c_[np.ones(X.shape[0]), X]
            prob = sigmoid(np.dot(X, self.w))
        else:
            prob = sigmoid(np.dot(self.X, self.w))
        return np.where(prob>=self.threshold, 1, 0).reshape((len(prob),))
    
    def plot(self, save_path):
        plt.figure()
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(1, 1)
        x1min, x1max = np.min(self.X[:,1]), np.max(self.X[:,1])
        ax.scatter(self.X[self.y==0, 1], self.X[self.y==0, 2], c='c', label='y=0')
        ax.scatter(self.X[self.y==1, 1], self.X[self.y==1, 2], c='b', label='y=1')
        x_line = np.arange(x1min, x1max, 0.001)
        y_line = -(self.w[0] + self.w[1]*x_line)/self.w[2]
        ax.plot(x_line, y_line, color='orange')
        ax.legend()
        plt.savefig(save_path)
    
  
       