import theano
import numpy
import theano.tensor as T

# 
theano.config.compute_test_value = "off"

# config shared variables
W1val = numpy.random.rand(2, 10, 10).astype(theano.config.floatX)
# declare a exist actual number or matrix as a shared value
W1 = theano.shared(W1val, 'W1');

W2val = numpy.random.rand(15, 20).astype(theano.config.floatX)
W2 = theano.shared(W2val, 'W2');

# input which will be of shape (5, 10)
x = T.matrix('x')

# transform the shared variable in some way. Theano does not
# know off hand that the matrix func_of_W1 has shape (20, 10)
func_of_W1 = W1.dimshuffle(2, 0, 1).flatten(2).T

# source of error: dot product of 5x10 with 20x10
h1 = T.dot(x, func_of_W1)

# do more stuff
h2 = T.dot(h1, W2.T)

# compile and call the actual function
f = theano.function([x], h2)
print f(numpy.random.rand(5, 10))