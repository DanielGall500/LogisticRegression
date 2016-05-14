import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

features = iris.data
target = iris.target

rand_num_gen = np.random

x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.3)

print('Number of features: ', len(features.T))

#placeholder dataset for testing
training_size = 400
num_features = 700

#D = (rand_num_gen.randn(training_size, features), rand_num_gen.randint(size=num_features, low=0, high=2))

training_steps = 10000

X = T.fmatrix('x')
Y = T.vector('y')

#The weight and bias are shared to keep their value between training iterations

weights = theano.shared(rand_num_gen.randn(len(features.T)), name='weights')

bias = theano.shared(0.0, name='bias')

Z = T.dot(X, weights) - bias

log_reg = 1 / (1 + T.exp(-Z))

predictions = log_reg > 0.5

cross_entropy = -(Y * T.log(log_reg) + (1 - Y) * T.log(1 - log_reg))

cost_function = cross_entropy.mean() + 0.01 * (weights ** 2).sum()

gradient_weights, gradient_bias = T.grad(cost_function, [weights, bias])

train = theano.function(inputs=[X,Y], outputs=[predictions, cross_entropy],
                        updates=((weights, weights * gradient_weights), (bias, bias * gradient_bias)), allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=[predictions])

for i in range(training_steps):
    predictions, error = train(features, target)

print('Y TEST', y_test)
print('PREDICTIONS', predictions)
print(accuracy_score(y_test, predictions))

#testing_pred = predict()

#accuracy = accuracy_score(target, testing_pred)


#print ('Accuracy Score: ', )























