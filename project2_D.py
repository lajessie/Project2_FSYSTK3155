"""Project 2 on Machine Learning,
    FSY-STK3155
    PART D) REGRESSION ANALYSIS OF THE ONE-DIMENSIONAL ISING MODEL USING NEURAL NETWORKS"""
    
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

class Neural_Network:
    def __init__(self,  X_dat, Y_dat,epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0):
    #parameters
        self.X_data_full = X_dat
        self.Y_data_full = Y_dat
        
        self.inputSize, self.n_features = X_dat.shape
        self.outputSize = 10
        self.hiddenSize = 50
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.inputSize // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
    
        #weights
        self.W1 = np.random.randn(self.n_features, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
        
        #bias
        self.B1 = np.zeros(self.hiddenSize) + .01 # (3x2) bias matrix from input to hidden layer
        self.B2 = np.zeros(self.outputSize) + .01 # (3x1) bias matrix from hidden to output layer
    
        self.weights = [self.W1, self.W2]
        self.biases = [self.B1, self.B2]

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
    #derivative of sigmoid
        return s * (1 - s)

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_dat, self.W1) + self.B1
        self.a_h = self.sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.W2) + self.B2
        self.probabilities = self.sigmoid(self.z_o)
        exp_term = np.exp(self.z_o)
        
        np.seterr(divide='ignore', invalid='ignore')
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
      # feed-forward for output
      z_h = np.matmul(X, self.W1) + self.B1
      a_h = self.sigmoid(z_h)
      
      z_o = np.matmul(a_h, self.W2) + self.B2
      exp_term = np.exp(z_o)
      
      probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
      return probabilities

    def backpropagation(self):
   
        # error in the output layer
        error_output = self.probabilities - self.Y_dat[0]
        # error in the hidden layer
        error_hidden = np.matmul(error_output, self.W2.T) * self.a_h * (1 -self.a_h)
        
        # gradients for the output layer
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        
        # gradient for the hidden layer
        self.hidden_weights_gradient = np.matmul(self.X_dat.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)
        
        if self.lmbd > 0.0:
                self.output_weights_gradient += self.lmbd * self.W2
                self.hidden_weights_gradient += self.lmbd * self.W1
    
        self.W2 -= self.eta * self.output_weights_gradient
        self.B2 -= self.eta * self.output_bias_gradient
        self.W1 -= self.eta * self.hidden_weights_gradient
        self.B1 -= self.eta * self.hidden_bias_gradient

    
    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)


    def train(self):
        data_indices = np.arange(self.inputSize)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_dat = self.X_data_full[chosen_datapoints]
                self.Y_dat = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


epochs = 100
batch_size = 100
eta = 0.01 #learning rate
lmbd = 0.01

dnn = Neural_Network(x_train, y_train, epochs=epochs, batch_size=batch_size, eta=eta,  lmbd=lmbd)
    
dnn.train()
test_predict = dnn.predict(x_test)


# equivalent in numpy
def accuracy_score(y_test, Y_pred):
    return np.sum(y_test == Y_pred) / len(y_test)

print("Accuracy score on test set: ", accuracy_score(y_test, test_predict))


