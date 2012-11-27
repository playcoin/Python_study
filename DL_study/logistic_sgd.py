import theano
import theano.tensor as T
import numpy
import cPickle, gzip

# load the data sets
# data_file = gzip.open('mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = cPickle.load(data_file)
# data_file.close()

class LogisticRegression(object):

	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(value=numpy.zeros((n_in, n_out)), dtype=theano.config.floatX, name='W')
		self.b = theano.shared(value=numpy.zeros((n_out,)), dtype=theano.config.floatX, name='b')

		# symbolic expression for computing the vector of class-membership probabilities
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)	# Theano has softmax API

		# symbolic desciption of how to compute prediction as class whose probability is maximal
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

	def negative_log_likelihood(self, y):
		# NLL loss function
		return - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

# complied Theano function that returns the vector of class-membership probabilities
# get_p_y_given_x = theano.function(inputs=[x], outputs=p_y_given_x)

# print the probability of some example represented by x_value
# x_value is not a symbolic variable but a numpy array describing the datapoint
# print 'probability that x is of class %i is %f' % (i, get_p_y_given_x(x_value)[i])
# the variable x_value and i is undefined?


# generate symbolic varaibles for input (x and y represent a minibatch)
x = T.fmatrix('x')
y = T.lvector('y')

# compiled theano function that returns this value
classifier = LogisticRegression(input=x.reshape(batch_size, 28*28), n_in=28*28, n_out=10)

cost = classifier.negative_log_likelihood(y)

# compute the gradient
g_W = T.grad(cost, classifier.W)
g_b = T.grad(cost, classifier.b)

# specify how to update hte parameters of the model as a dictionary
update = {classifier.W : classifier.W - learning_rate * g_W,\
	classifier.b : classifier.b - learning_rate * g_b}

train_model = theano.function(inputs=[index],
	outputs = cost,
	updates = updates,
	givens = {
		x: train_set_x[index * batch_size: (index + 1) * batch_size],
		y: train_set_y[index * batch_size: (index + 1) * batch_size]
	})

