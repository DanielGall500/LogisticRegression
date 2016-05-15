import theano
import theano.tensor as T
import numpy as np

rand_num_gen = np.random

training_steps = 10000

X = T.matrix('x')
Y = T.vector('y')

weights = theano.shared(rand_num_gen.randn(len(features.T)), name='weights')

bias = theano.shared(0.0, name='bias')

Z = theano.dot(X, weights) - bias

log_reg = 1 / (1 + T.exp(-Z))

predictions = log_reg > 0.5

cross_entropy = -(Y * T.log(log_reg) + (1 - Y) * T.log(1 - log_reg))

cost_function = cross_entropy.mean() + 0.01 * (weights ** 2).sum()

gradient_weights, gradient_bias = T.grad(cost_function, [weights, bias])

train = theano.function(inputs=[X,Y], outputs=[predictions, cross_entropy], updates=((weights, weights * gradient_weights), (bias, bias * gradient_bias)))

predict = theano.function(inputs=[X], outputs=[predictions])

for i in range(training_steps):
    predictions, error = train(x_train, y_train) #Your own data input























