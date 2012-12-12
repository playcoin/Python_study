import numpy
import theano
import theano.tensor as T

a = theano.shared(1)
b = theano.shared(5)
x = T.iscalar('x')

# use lambda expression as fn
# fn = lambda : {a : a + 1}

# use defined function
def fn(x):
	return x, {a : a + 1 + x, b : b * 3}	# this form is more comprehensive than the lambda 
											# Ok, It's obvious now, the scan function return a tuple: (return_values, updates)
											# the return_values will be used in loops. The updates is dict type, will store the
											# update value for current loop. 

# the ups is a Dictionary type
values, ups = theano.scan(fn = fn, outputs_info=x, n_steps=10)
print ups

c = ups[a] + 5

d = ups[b]

f = theano.function([x], [c, d], updates=ups)

print f(3)