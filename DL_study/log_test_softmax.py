import theano
import theano.tensor as T
import numpy

# Test a 10 sample logistic regression
# the range of y is 0~2
x = T.fmatrix('x')
y = T.lvector('y')

# parameters. W and b can not be initialized by numpy.random, Because the each column of the matrix W should be the same.
b = theano.shared(value=numpy.zeros((3,), dtype=theano.config.floatX), name="b")
W = theano.shared(value=numpy.zeros((2, 3), dtype=theano.config.floatX), name="W")

# softmax
p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)

# compiled function
get_p_y_given_x = theano.function([x], p_y_given_x)

a = numpy.asmatrix([[0.,1],
	[1,2],
	[2,3],
	[3,4],
	[4,5],
	[5,6],
	[6,7],
	[7,8],
	[8,9],
	[9,10]
	], dtype="float32")

ca = numpy.asarray([0,1,2,1,2,1,1,0,2,2], dtype='int32')

print "Test the conditional probabilites: " , get_p_y_given_x(a)

# check W and b
print "Print the parameters: "
print "W: " , W.get_value()
print "b:" , b.get_value()

# do argmax
y_pred = T.argmax(p_y_given_x, axis=1)
# print theano.pp(y_pred)

# compile classify function
classify = theano.function([x], y_pred)

print classify(a)

loss = - T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

f_nll = theano.function([x, y], loss)

print f_nll(a, ca)