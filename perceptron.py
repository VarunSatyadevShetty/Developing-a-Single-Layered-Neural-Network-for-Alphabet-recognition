import numpy as np
np.random.seed(42)

##NOTE we are doing X(which is a [1,2] matrix) multiplied by W(which is a [2,1] matrix) + b which is a scalar

#this function calculates a step function also called as heaviside function
def step_function(value):
    if(value>=0):
        return 1
    else:
        return 0

#this function returns the prediction value
def prediction(W,X,b):
    value = np.matmul(X,W) + b
    return (step_function(b))

#this is the perceptron function i.e in this function we will be calculating the weights and the bias
def perceptron(W,X,b,y,learning_rate):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if(y[i]-y-hat==1):               #this is the case when we prdicted it false and the actual value is true
            W[0] = W[0] + (learning_rate * X[i][0])
            W[1] = W[1] + (learning_rate * X[i][1])
            b = b + (learning_rate * 1)
        elif(y[i]-y_hat==-1):            #this is the case when the predicted value is true and the actual value is false
            W[0] = W[0] - (learning_rate * X[i][0])
            W[1] = W[1] - (learning_rate * X[i][1])
            b = b - (learning_rate * 1)
    return W,b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes
def training_perceptron_algorithm(X,y,learning_rate=0.1,number_of_epoch=25):
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)
    boundary_lines = []
    for i in range(number_of_epoch):
        W,b = perceptron(W,X,b,learning_rate)
        boundary_lines.append((-W[0]/W[1],-b/W[1]))
    return boundary_lines
