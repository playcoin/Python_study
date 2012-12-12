import numpy
import theano
import theano.tensor as T
from theano import function

# test the multiple outputs

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff * 2
f = function([a, b], [diff, abs_diff, diff_squared])

out = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

print theano.pp(diff)
print out

m = numpy.asarray([[1, 2], [3, 4]])
print m[0]
print m[1]
print m[0][1]
print m[1][1]
print m.get_row()