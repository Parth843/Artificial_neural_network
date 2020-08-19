import numpy as np
import scipy.special
import pandas as pd
from sklearn.model_selection import train_test_split

class neuralNetwork:
    def __init__(self,l1nodes,l2nodes,l3nodes,learning_rate):
        
        # Define number of nodes in each layer including bias if used 
        
        self.l1 = l1nodes  
        self.l2 = l2nodes 
        self.l3 = l3nodes
        
        # Initialize weights (bias nodes do not get any inputs) 
        
        self.theta1 = np.random.normal(0.0,pow(self.l1,-0.5),((self.l2 - 1), self.l1))
        self.theta2 = np.random.normal(0.0,pow(self.l2,-0.5),(self.l3, self.l2))
        
        # Define learning rates
        
        self.lr = learning_rate
        
        # Define activation function (sigmoid or tanh)
        
        self.af = lambda x : scipy.special.expit(x)
    
    def train(self,inputs,targetlist):
        # Convert input and target list to numpy arrays
        inputs = np.array(inputs,ndmin = 2).T
        targets = np.array(targetlist,ndmin = 2).T
        
        # Calculate the layer2 inputs and outputs
        l2ip = np.dot(self.theta1,inputs)
        l2op = self.af(l2ip)
        # Add bias node to the output of layer2
        bias = np.ones([1,l2op.shape[1]])
        l2op_biased = np.vstack((bias,l2op))
        
        # Calculate input and output of layer3 (which is the output layer)
        l3ip = np.dot(self.theta2,l2op_biased)
        l3op = self.af(l3ip)
        
        # Calculate error in layer3 for backprop
        errorl3 = l3op - targets
        # Calculate error in layer2
        errorl2 = np.dot(self.theta2.T,errorl3)
        # Delete the first row of errorl2 which represents error in bias
        errorl2 = np.delete(errorl2,0,0)
        
        # Update weights
        self.theta2 -= self.lr*np.dot((errorl3*l3op*(1-l3op)),np.transpose(l2op_biased))
        self.theta1 -= self.lr*np.dot((errorl2*l2op*(1-l2op)),np.transpose(inputs))
        
    
    def query(self,inputs):
        # Calculate the output for a given input
        inputs = np.array(inputs, ndmin = 2).T
        
        l2ip = np.dot(self.theta1,inputs)
        l2op = self.af(l2ip)
        bias = np.ones([1,l2op.shape[1]])
        l2op_biased = np.vstack((bias,l2op))
        
        l3ip = np.dot(self.theta2 , l2op_biased)
        l3op = self.af(l3ip)
        
        return l3op
        
        

data = pd.read_csv('digit_recogniser_short.csv') # Import the data 

y = data['label'] # Set the label column as the target

data = train_data.drop(['label','Unnamed: 0'], axis = 1)# Remove the target column from data 

# Scale the data to range 0.99 to 0.001
train_data = train_data/255 * 0.99 + 0.001

# Convert data to numpy array
X = train_data.values

# Split data into train and test datasets
xtrain, xtest, ytrain, ytest = train_test_split(X,y,stratify = y, test_size = 0.2,random_state = 1)

# Create a neuralNetwork instance 
n1 = neuralNetwork(785, 200, 10, 0.4)

# Train the neural network for 4 epoch

for epoch in range(4):
    for i in range(len(xtrain)):
        inputs = [1] + list(xtrain[i])
        targets = np.zeros(10) + 0.001
        targets[ytrain.iloc[i]] = 0.99
        n1.train(inputs, targets)

# Evaluate the neural network based on accuracy

score = []

for i in range(len(xtest)):
    testip = [1] + list(xtest[i])
    prediction = n1.query(testip).argmax()
    result = prediction == ytest.iloc[i]
    score.append(result)
    
np.mean(score)

# 93% accuracy on kaggle.