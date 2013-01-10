import numpy
import theano
import theano.tensor as T
import cPickle, gzip, os, sys, time

from logistic_sgd import LogisticRegression, load_data

# define the hiddenlayer class
class HiddenLayer(object):

	# init function
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		"""
		Typical hidden layer of a MLP: units are fully-connected and have
		sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).

		NOTE : The nonlinearity used here is tanh

		Hidden unit activation is given by: tanh(dot(input,W) + b)

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dmatrix
		:param input: a symbolic tensor of shape (n_examples, n_in)

		:type n_in: int
		:param n_in: dimensionality of input

		:type n_out: int
		:param n_out: number of hidden units

		:type activation: theano.Op or function
		:param activation: Non linearity to be applied in the hidden
						      layer
		"""
		self.input = input
		# init W and b
		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low  = -numpy.sqrt(6.0 / (n_in + n_out)),
				high  = numpy.sqrt(6.0 / (n_in + n_out)),
				size = (n_in, n_out)), dtype = theano.config.floatX
			)
			# if the activation is sigmoid, the W_values should multiply 4
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value = W_values, name='W', borrow=True)
		
		if b is None:
			b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
			b = theano.shared(value = b_values, name='b', borrow=True)

		# assign to shared symbols
		self.W = W
		self.b = b

		lin_out = T.dot(input, self.W) + self.b
		# assign the theano expression to the attribute 'output' of the HiddenLayer
		self.output = (lin_out if activation is None else activation(lin_out))
		# 'output' is a theano expression, it should be pass to the later layer,
		# so that all the expression can be compiled together.

		# parameters
		self.params = [self.W, self.b]

# the MLP class, contains a HiddenLayer and a LogisticRegression Layer
class MLP(object):

	def __init__(self, rng, input, n_in, n_hidden, n_out):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
		architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
		which the datapoints lie

		:type n_hidden: int
		:param n_hidden: number of hidden units

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
		which the labels lie

		"""
		self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in = n_in,
										n_out = n_hidden, activation = T.tanh)
		# the log regresion layer
		self.logReressionLayer = LogisticRegression(input=self.hiddenLayer.output,
												n_in = n_hidden,
												n_out = n_out)
		# L1 norm
		self.L1 = abs(self.hiddenLayer.W).sum() \
				+ abs(self.logReressionLayer.W).sum()

		# L2 norm
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
				+ (self.logReressionLayer.W ** 2).sum()

		# negative log likelihood
		# self.logReressionLayer.negative_log_likelihood is a python function
		self.negative_log_likelihood = self.logReressionLayer.negative_log_likelihood
		# the negative_log_likelihood function is defined in the class LogisticRegression,
		# so the MLP can use the function directly

		self.errors = self.logReressionLayer.errors

		# the params
		self.params = self.hiddenLayer.params + self.logReressionLayer.params

# actual test procedure
def test_mlp(learning_rate = 0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs = 1000,
			dataset="./data/mnist.pkl.gz", batch_size=20, n_hidden=500):
	"""
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
				 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


	"""
	
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# computer number of minibathces for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
	print "batches, train: %d, valid: %d, n_test_batches: %d" % (n_train_batches, n_valid_batches, n_test_batches)	

	######################
	# Build Actual Model #
	######################
	print '... build the model'

	# generate symbolic varaibles for input (x and y represent a minibatch)
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	rng = numpy.random.RandomState(1234)	# 1234? seed?

	# contruct a MLP object
	classifier = MLP(rng = rng, input=x, n_in=28*28, n_hidden=n_hidden, n_out=10)

	# cost with L1 and L2 norm
	cost = classifier.negative_log_likelihood(y) \
		+ L1_reg * classifier.L1 \
		+ L2_reg * classifier.L2_sqr


	# function for computes the mistakes that are made by the model on a minibatch
	test_model = theano.function(inputs=[index],
		outputs = classifier.errors(y),
		givens = {
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		})

	validation_model = theano.function(inputs=[index],
		outputs = classifier.errors(y),
		givens = {
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		})

	# the update strategies is different from the sigle logistic regression

	# compute the gradient of cost with respect to theta
	gparams = []
	for param in classifier.params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)
	# it seems not to be an effient approach to compute the gradients
	# the backpropagation will be faster ?

	updates = {}

	# user the zip operator
	for param, gparam in zip(classifier.params, gparams):
		updates[param] = param - learning_rate * gparam

	# function for training one batch
	train_model = theano.function(inputs=[index],
		outputs = cost,
		updates = updates,
		givens = {
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		})
	###############
	# Train Model #
	###############	
	# consider how to use the batches

	# early-stopping parameters
	patience = 10000
	patience_increase = 2
	improvement_thereshold = 0.995
	validation_frequency = min(n_train_batches, patience / 2)
	print "The frequency of validation is :" , validation_frequency

	best_params = None
	best_validation_loss = numpy.inf
	test_score = 0.
	best_iter = 0
	start_time = time.clock()

	done_looping = False
	epoch = 0
	while (epoch < n_epochs) and (not done_looping):
		epoch += 1

		for minibatch_index in xrange(n_train_batches):

			minibatch_avg_cost = train_model(minibatch_index)
			# iteration number
			c_iter = epoch * n_train_batches + minibatch_index

			if (c_iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validation_model(i) 
									for i in xrange(n_valid_batches)]
				this_validation_lost = numpy.mean(validation_losses)

				print ('epoch %i, minibatch %i/%i, validation error %f %%' % \
					(epoch, minibatch_index + 1, n_train_batches, 
						this_validation_lost * 100.))

				# if we got the best validation score untial now
				if this_validation_lost < best_validation_loss:
					# improve patience if loss improvement is good enough
					if this_validation_lost < best_validation_loss * \
						improvement_thereshold:
						patience = max(patience, c_iter * patience_increase)

					best_validation_loss = this_validation_lost
					best_iter = c_iter
					# test it on the test set

					test_losses = [test_model(i)
									for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(('     epoch %i, minibatch %i/%i, test error of best'
				       ' model %f %%') %
						(epoch, minibatch_index + 1, n_train_batches,
						 test_score * 100.))

			if patience <= c_iter: # if patience <= c_iter , then means the validation_loss has not been optimized for a long time
				done_looping = True
				break

	# still in while loop
	end_time = time.clock()
	print(('Optimization complete. Best validation score of %f %% '
			'obtained at iteration %i, with test performance %f %%') %
			(best_validation_loss * 100., best_iter, test_score * 100.))

	print 'The code run for %d epochs, with %f epochs/sec' % (
		epoch, 1. * epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' + 
						os.path.split(__file__)[1] +
						' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
	test_mlp()




