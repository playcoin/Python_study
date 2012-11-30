import theano
import theano.tensor as T
import numpy
import cPickle, gzip, os, sys, time

class LogisticRegression(object):

	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
		self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b')

		# symbolic expression for computing the vector of class-membership probabilities
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)	# Theano has softmax API

		# symbolic desciption of how to compute prediction as class whose probability is maximal
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

		# parameters
		self.params = [self.W, self.b];


	def negative_log_likelihood(self, y):
		# NLL loss function
		return - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
		over the total number of examples of the minibatch; zero one loss over the 
		size of the minibatch
		"""
		# check the dimension
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y shoul have the same shape as self.y_pred',
				('y', target.type, 'y_pred', self.y_pred.type))
		# check the datatype
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

# define the load_data function
# return 3 part data for train, valid, test set
def load_data(dataset):

	# Load the Minist dataset from local machine
	data_dir, data_file = os.path.split(dataset)
	if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
		raise EOFError("Dataset doesn't exists!")

	print '... loading data'

	# load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	# define a function in a function!!
	def shared_dataset(data_xy, borrow=True):
		"""The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
		"""

		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
								borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
								borrow=borrow)

		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), 
		(test_set_x, test_set_y)]

	return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
							dataset="./data/mnist.pkl.gz",
							batch_size = 600):
	
	"""
	Demonstrate stochastic gradient descent optimization of a log-linear model
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

	# construct the logistic regression class
	# Each MNIST image has size 28*28
	classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

	# the cost
	cost = classifier.negative_log_likelihood(y)

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

	# compute the gradients
	g_W = T.grad(cost, classifier.W)
	g_b = T.grad(cost, classifier.b)

	# specify how to update the parameters of the model as a dictionary
	updates = {classifier.W : classifier.W - learning_rate * g_W,\
		classifier.b : classifier.b - learning_rate * g_b}

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
	patience = 5000
	patience_increase = 2
	improvement_thereshold = 0.995
	validation_frequency = min(n_train_batches, patience / 2)
	print "The frequency of validation is :" , validation_frequency

	best_params = None
	best_validation_loss = numpy.inf
	test_score = 0.
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
					# test it on the test set

					test_losses = [test_model(i)
									for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

					print "     Parameter W[0] : " , classifier.W.get_value()[154] , "\n"
					print "     Parameter b : " , classifier.b.get_value() , "\n"

			if patience <= c_iter: # if patience <= c_iter , then means the validation_loss has not been optimized for a long time
				done_looping = True
				break

	# still in while loop
	end_time = time.clock()
	print(('Optimization complete with best validation score of %f %%,'
		'with test performance %f %%') %
			(best_validation_loss * 100., test_score * 100.))

	print 'The code run for %d epochs, with %f epochs/sec' % (
		epoch, 1. * epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' + 
						os.path.split(__file__)[1] +
						' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
	sgd_optimization_mnist()