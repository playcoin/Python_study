import numpy
import theano
import theano.tensor as T
from mlp_test import HiddenLayer

class SquareErrors(object):
	"""	The minimum square errors function
	"""

	def __init__(self, input, n_in, W=None, b=None):
		""" init the W and b from parameters """
		if W is None:
			W = theano.shared(value=numpy.zeros((n_in, 1), dtype=theano.config.floatX), name='W')

		if b is None:
			b = theano.shared(value=numpy.zeros((1,), dtype=theano.config.floatX), name='b')

		self.W = W
		self.b = b
		# predict
		self.pred = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

		self.params = [self.W, self.b]

def test_hls(learning_rate = 0.01):
	""" Test a simple 2 hidden layer NN
	"""
	x = T.vector('x')
	y = T.scalar('y')

	rng = numpy.random.RandomState(1234)	# 1234? seed?

	# The HiddenLayer is feasible enough to initialize the parameters by myself.
	# Be careful with the row and col line. Use the multiply operator between vector and matrix.
	l1W = theano.shared(numpy.asarray([[2,1], [-2,3]], dtype=theano.config.floatX))
	l1b = theano.shared(numpy.asarray([0,-1], dtype=theano.config.floatX))
	layer1 = HiddenLayer(rng = rng, input=x, n_in = 2, n_out = 2,  activation=T.nnet.sigmoid, W = l1W, b = l1b)

	l2W = theano.shared(numpy.asarray([3,-2], dtype=theano.config.floatX))
	l2b = theano.shared(numpy.asarray([-1,], dtype=theano.config.floatX))
	layer2 = SquareErrors(input=layer1.output, n_in=2, W=l2W, b=l2b)

	pred = layer2.pred

	# The minimum square errors
	errors = cost = (y - pred[0]) ** 2

	hparams = layer1.params + layer2.params

	# update the parameters
	gparams = []
	for param in hparams:
		gparam = T.grad(cost, param)
		gparams.append(gparam)
	# it seems not to be an effient approach to compute the gradients
	# the backpropagation will be faster ?

	updates = {}
	# use the zip operator
	for param, gparam in zip(hparams, gparams):
		updates[param] = param - learning_rate * gparam

	# train function
	print "Compiling the functions...."
	train_model = theano.function(inputs=[x, y], outputs=[pred], updates=updates)
	print "Compiling Over!"

	printParams(0, 0, layer1, layer2)

	pred_v = train_model([1, 0], 0)
	printParams(1, pred_v, layer1, layer2)

	pred_v = train_model([0, 0], 1)
	printParams(2, pred_v, layer1, layer2)

	pred_v = train_model([0, 1], 0)
	printParams(3, pred_v, layer1, layer2)

	pred_v = train_model([1, 1], 1)
	printParams(4, pred_v, layer1, layer2)

def printParams(rnd, pred, layer1, layer2):
	print "The pred is :", pred
	print "Round %d for W and b of the layer1" % rnd
	print "W1 is:\n%s" % layer1.W.get_value().T
	print "b1 is:\n%s" % layer1.b.get_value()

	print "Round %d for W and b of the layer2" % rnd
	print "W2 is:\n%s" % layer2.W.get_value().T
	print "b2 is:\n%s\n" % layer2.b.get_value()

if __name__ == "__main__":
	test_hls(0.5)	# in back propagation, the coefficient number 2 is fuse into the variable 'c'.
					# So learning rate 0.5 here is equivalent to 1.0 in back propagation algorithm.
