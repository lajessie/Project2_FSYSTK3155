"""Project 2 on Machine Learning,
    FSY-STK3155 """
    
from sklearn.cross_validation import train_test_split 
import scipy.linalg as scl
from tabulate import tabulate
import numpy as np

np.random.seed(12)

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

def ols_regression(data_train, data_test, depen):
    """ Function that performs the OLS regression with the inverse"""
    beta_ols = np.linalg.inv(data_train.T @ data_train) @ data_train.T @ depen
    pred= data_test@beta_ols
    
    return beta_ols, pred


def ols_SVD(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Function that performs the OLS regression with the Singular Value Descomposition"""
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y

def ridge_regression(data_train, data_test, depen, alpha):
    """Funtions that performs the Ridge Regression"""
    beta_olsRidge = np.linalg.inv(data_train.T @ data_train + alpha*np.identity(1600)) @ data_train.T @ depen
    pred= data_test@beta_olsRidge
    return beta_olsRidge, pred

"""The following functions calculate different measurements to assess the model"""
def mse(y_pred, y_test):
    return np.mean( np.mean((y_test - y_pred)**2) )

def r2(y_pred, y_test):
    return np.sqrt(np.mean( np.mean((y_test - y_pred)**2) ))

def bias(y_pred, y_test):
    return np.mean( (y_test - np.mean(y_pred))**2 )

def variance(y_pred, y_test):
    return np.mean( np.var(y_pred)) 

def all_values(y_pred, y_test):
    measuremnts = [['MSE', mse(y_pred,y_test)],
         ['R2', r2(y_pred,y_test)],
         ['Bias', bias(y_pred,y_test)],
         ['Variance', variance(y_pred,y_test)]]
    return measuremnts
 
""" Define Ising model """
#system size
L = 40
#create 1000 random Ising states
states = np.random.choice([-1, 1], size=(1000, L))

# calculate Ising energies
energies=ising_energies(states,L)  # Y or dependent var

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2])) # X or independent var

#Split the data into train and test (Using Cross-Validation from sklearn)
x_train, x_test, y_train, y_test = train_test_split(states, energies, test_size=0.2)


"""It's not possible perform ols with the function ols_regression 
because the matrix (data.T@data) is singular and it's not possible compute the inverse """
#Beta_OLS, pred_OLS = ols_regression(x_train, x_test, y_train)
"""Istead, the Singular Value descomposition is used"""

coefs_OLS = ols_SVD(x_train, y_train)
y_pred= x_test@coefs_OLS
#Print results
print('\n--OLS with SVD--')
print(tabulate(all_values(y_pred, y_test)))


#-------Trying with the OLS in sklearn-----------
from sklearn.linear_model import LinearRegression
regrGr = LinearRegression()
regrGr.fit(x_train, y_train)
betaSK = regrGr.coef_
predSK = regrGr.predict(x_test)
#Print results
print('\n--OLS with sklear--')
print(tabulate(all_values(predSK, y_test)))


#--------------Ridge Regression--------------
alphas= np.logspace(-4, 5, 10)
MSE_R = 1
for val in alphas:
    beta, prediRidge = ridge_regression(x_train, x_test, y_train, val)
    mseRidge = mse(prediRidge, y_test)
    if (mseRidge <= MSE_R):
        alphaR= val
        Beta_R= beta
        pred_Ridge = prediRidge
        MSE_R = mseRidge
        
#Print results       
print('\n--Ridge regression--')
print('The best value for alpha = ', alphaR)  
print(tabulate(all_values(pred_Ridge,y_test)))    
        

#--------------Lasso Regression-------------
from sklearn.linear_model import Lasso #Using Sklearn as I did in 1st project
alphasL= np.logspace(-4, 5, 10)
regr = Lasso()
scores = [regr.set_params(alpha = alpha).fit(x_train, y_train).score(x_test, y_test) for alpha in alphasL]
best_alpha = alphasL[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(x_train, y_train)
pred_Lasso = regr.predict(x_test)

#Print results       
print('\n--Lasso regression--')
print('The best value for alpha = ', best_alpha)  
print(tabulate(all_values(pred_Lasso,y_test))) 



