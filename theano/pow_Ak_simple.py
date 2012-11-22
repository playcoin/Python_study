import numpy

def power(A, k) :
	result = 1
	for i in range(k):
		result = result * A

	return result


print power(numpy.asarray([1,2,3]), 2)