"""
Project 2 on Machine Learning,
    FSY-STK3155
    PART C)
"""
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.model_selection as skms
import pickle,os, glob

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        #Standard Gradient Descent
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
              
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
    

#Ising model
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
#Read the files for the 2D data
filenames = glob.glob(os.path.join("..", "dat", "*"))
label_filename = "Ising2DFM_reSample_L40_T=All_labels.pkl"
dat_filename = "Ising2DFM_reSample_L40_T=All.pkl"
file_name = "Ising2DFM_reSample_L40_T=All.pkl"
# Read in the labels
with open(label_filename, "rb") as f:
    labels = pickle.load(f)

# Read in the corresponding configurations
with open(dat_filename, "rb") as f:
    data = np.unpackbits(pickle.load(f)).reshape(-1, 1600).astype("int")

# Set spin-down to -1
data[data == 0] = -1


# Set up slices of the dataset
ordered = slice(0, 70000)
critical = slice(70000, 100000)
disordered = slice(100000, 160000)

X_train, X_test, Y_train, Y_test = skms.train_test_split(
    np.concatenate((data[ordered], data[disordered])),
    np.concatenate((labels[ordered], labels[disordered])),
    test_size=0.95
)

model = LogisticRegression(lr=0.1, num_iter=100000)
model.fit(X_train,Y_train)
preds = model.predict(X_test)

accuracy = (preds == Y_test).mean()
print(accuracy)
print(model.theta)

#Compare with Sklearn
from sklearn.linear_model import LogisticRegression

modelS = LogisticRegression(C=1e20)
modelS.fit(X_train, Y_train)
preds = modelS.predict(X_test)
accuracyS = (preds == Y_test).mean()
print(accuracyS)
print(modelS.intercept_, modelS.coef_)
