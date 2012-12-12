import numpy
import theano
import theano.tensor as T
from theano import function
from theano import pp
rng = numpy.random

x = T.dscalar('x')
y = x ** 2z = T.grad(y, x)
f = function([x], y)
g = function([x], z)

# print the gradient function by theano.pp
print pp(z)

print g(4)

print g(92.3)

# Test dlogistic

x = T.dmatrix("x")
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
# change the compile mode
dlogistic = function(inputs=[x], outputs=gs, mode='FAST_RUN')
randMat = rng.randn(10, 4)
print randMat
print dlogistic(randMat)