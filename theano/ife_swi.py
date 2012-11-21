# #from theano import tensor as T
import theano.tensor as T
# import theano.ifelse as ifelse
from theano import function
from theano.ifelse import ifelse # It has to write in the manner. 'ifelse' is not a attribute of theano.ifelse
import theano, time, numpy

# test ifelse and switch
a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')

# switch seems strange
z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
# ifelse
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = function([a, b, x, y], z_switch, mode=theano.Mode(linker="vm"))
f_lazy = function([a, b, x, y], z_lazy, mode=theano.Mode(linker="vm"))

val0 = 0
val1 = 1
big_mat1 = numpy.ones((10000, 1000))
big_mat2 = numpy.ones((10000, 1000))

n_times = 10;

tic = time.clock()
for i in range(n_times):
	f_switch(val0, val1, big_mat1, big_mat2)

print 'time spent evaluating both values %f sec' % (time.clock() - tic)

tic = time.clock()
for i in range(n_times):
	f_lazy(val0, val1, big_mat1, big_mat2)

print 'time spent evaluating both values %f sec' % (time.clock() - tic)
