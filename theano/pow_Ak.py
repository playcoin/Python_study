import theano
import theano.tensor as T

theano.config.warn.subtensor_merge_bug = False

k = T.iscalar('k')
A = T.vector('A')

# define a function
def inner_fct(prior_result, A):
	return prior_result * A

# symbolic description of the result
result, updates = theano.scan(fn=inner_fct, outputs_info=T.ones_like(A), non_sequenceS=A, n_steps=k)

final_result = result[-1];

power = theano.function(inputs=[A, k], outputs=final_result, updates=updates)

print power(range(10), 2)
