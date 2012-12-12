import numpy
import theano
import theano.tensor as T
theano.config.warn.subtensor_merge_bug = False
theano.config.floatX = 'float32'

coefficients = T.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = T.arange(max_coefficients_supported)
# components, updates = theano.scan(fn=lambda coeff, power, free_var:
#                                   coeff * (free_var ** power),
#                                   sequences=[coefficients, full_range],
#                                   outputs_info=None,
#                                   non_sequences=x)
# polynomial = components.sum()

# a memory-efficient approach to calculate polynomial
'''The default python dtype of 0. is float32. theano's float64 can not involve an downcast.
One way to solve the problem is set the theano.config.floatX to 'float32'
Another way is wrap the python variable to theano variable.
For example: outputs_info = T.as_tensor_variable(numpy.asarray(0, x.dtype))'''
components, updates = theano.scan(fn=lambda coeff, power, poly_sum, free_var:
                                  poly_sum + coeff * (free_var ** power),
                                  sequences=[coefficients, full_range],
                                  outputs_info=0.,
                                  non_sequences=x)

polynomial = components[-1]
calculate_polynomial1 = theano.function(inputs=[coefficients, x],
                                        outputs=polynomial)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print calculate_polynomial1(test_coeff, 3)