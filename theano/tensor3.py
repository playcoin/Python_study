import theano
import theano.tensor as T
from theano import function

# test the shared variables

state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

print state.get_value()
print accumulator(1)
print state.get_value()
print accumulator(300)
print state.get_value()

# reset the state
state.set_value(-1)
print accumulator(3)
print state.get_value()

# use the shared variables everywhere
decrementor = function([inc], state, updates=[(state, state-inc)])
print decrementor(2)
print state.get_value()

# test givens
fn_of_state = state * 2 + inc

foo = T.iscalar()
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)
print state.get_value()
