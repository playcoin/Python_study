import theano
import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams(seed = 234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)
g1 = function([], rv_n)

nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
f_val0 = f()
print f_val0
print '###'
f_val1 = f()
print f_val1

print '\ncalling function g'
g_val0 = g()
g_val1 = g()
print g_val0
print '####'
print g_val1

print '\ncalling function g1'
g1_val0 = g1()
g1_val1 = g1()
print g1_val0
print '####'
print g1_val1

print '\nnearly zeros'

n_val = nearly_zeros()
print n_val

# reset rng 
print "\nReset rng"
state_after_v0 = rv_u.rng.get_value().get_state()
v1 = f();
print "v1", v1
v2 = f();
print "v2", v2
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0);
rv_u.rng.set_value(rng, borrow=True)
v3 = f()
print 'v3', v3