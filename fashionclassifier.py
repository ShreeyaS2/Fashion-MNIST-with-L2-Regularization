import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import softmax      

data=pd.read_csv('C:\\Personal\\comp_proj\\neuralnetwork\\fashion-mnist_train.csv')
m,n=data.shape
data=np.array(data) 
test_data=data[0:500].T #splitting into testing (0-1000) and training (1000-m)
Y_dev=test_data[0] #after transposing, seperating the label row
X_dev=test_data[1:n]/255.0 # splitting till n for test data and normalizing by dividing by 255
train_data=data[500:m].T # taking the training data from 1000 to m and transposing
Y_train=train_data[0] #seperating the label row
X_train=train_data[1:n]/255.0 # splitting till n for train data and normalizing by dividing by 255

#initializing parameters
def initialize_params():
    W1=np.random.rand(10,784) -0.5
    b1=np.zeros((10,1))
    W2=np.random.rand(10,10) -0.5
    b2=np.zeros((10,1))
    return W1,b1,W2,b2

# stepone: forward propagation
def forward_prop(W1,b1,W2,b2,X):
    Z1=np.dot(W1,X) +b1
    #replace sigmoid with relu
    A1=np.maximum(Z1,0) #relu activation function
    Z2=np.dot(W2,A1)+b2
    A2=softmax(Z2,axis=0) #softmax activation function
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def relu_derivative(Z):
    return Z > 0

# steptwo: backward propagation
def backward_prop(A1,A2,W1,W2,X,Y,Z1,l2_lambda):
    m=Y.size
    one_hot_Y=one_hot(Y)
    dZ2=A2-one_hot_Y
    dW2=1/m*np.dot(dZ2,A1.T) + (l2_lambda/m)*W2
    db2=1/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.dot(W2.T,dZ2)*relu_derivative(Z1)
    dW1=1/m*np.dot(dZ1,X.T) + (l2_lambda/m)*W1
    db1=1/m*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate):
    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2
    return W1,b1,W2,b2
    
def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations,l2_lambda):
    W1, b1, W2, b2 = initialize_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(A1, A2, W1, W2, X, Y, Z1, l2_lambda)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

# No regularization
W1_nr, b1_nr, W2_nr, b2_nr = gradient_descent(
    X_train, Y_train, alpha=0.05, iterations=1000, l2_lambda=0.0
)

# With L2 regularization
W1_l2, b1_l2, W2_l2, b2_l2 = gradient_descent(
    X_train, Y_train, alpha=0.05, iterations=1000, l2_lambda=0.1
)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_dev[:, index, None]
    prediction = make_predictions(X_dev[:, index, None], W1, b1, W2, b2)
    label = Y_dev[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
# Test predictions without regularization
test_prediction(0, W1_nr, b1_nr, W2_nr, b2_nr)
# Test predictions with L2 regularization
test_prediction(0, W1_l2, b1_l2, W2_l2, b2_l2)
