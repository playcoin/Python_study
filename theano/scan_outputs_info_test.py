import numpy
import theano
import theano.tensor as T

x = T.iscalar('x')

# use defined function
def fn(x):
	return [x*3, x, x+1, x*2]

# the ups is a Dictionary type
values, ups = theano.scan(fn = fn, outputs_info=[None,None,x,None], n_steps=10)

f = theano.function([x], values)

print f(3)

"""
    This kind of typing is pretty interesting!
    None is placeholder, x is the initial state corresponding to the 3rd output
"""