import numpy
import theano
import theano.tensor as tt
theano.config.warn.subtensor_merge_bug = False

coefficients = tt.vector("coefficients")
x = tt.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = tt.arange(max_coefficients_supported)
components, updates = theano.scan(fn=lambda coeff, power, free_var:
                                  coeff * (free_var ** power),
                                  sequences=[coefficients, full_range],
                                  outputs_info=None,
                                  non_sequences=x)
polynomial = components.sum()
calculate_polynomial1 = theano.function(inputs=[coefficients, x],
                                        outputs=polynomial)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print calculate_polynomial1(test_coeff, 3)