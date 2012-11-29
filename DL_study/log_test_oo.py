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

		print "###### LogisticRegression __init__ over! #########"

	def negative_log_likelihood(self, y):
		print "###### call LogisticRegression negative_log_likelihood! #########"
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

	# the cost function
	# actually the 'classifier'
	cost = classifier.negative_log_likelihood(y)
	print theano.printing.debugprint(cost)

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

	# test classifier's variable
	# train the first batch
	# train_model(0);
	# print "     Parameter W[0] : " , classifier.W.get_value()[154] , "\n"
	# print "     Parameter b : " , classifier.b.get_value() , "\n"
	# train_model(1);
	# print "     Parameter W[0] : " , classifier.W.get_value()[154] , "\n"
	# print "     Parameter b : " , classifier.b.get_value() , "\n"


if __name__ == '__main__':
	sgd_optimization_mnist()