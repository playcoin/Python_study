import numpy
import theano
import theano.tensor as T
from theano import function

# test the the logistic function
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
a = numpy.asarray([[0 ,1], [-1, -2]])
logistic = function([x], s)
print logistic(a)

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)
print logistic2(a)