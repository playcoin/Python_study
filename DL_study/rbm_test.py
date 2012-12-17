import cPickle
import gzip
import time
import PIL.Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data

# try to write RBM code
class RBM(object):
	""" Restricted Boltzmann Machine"""
	def __init__(self, input=None, n_visible=784, n_hidden=500,
		W = None, hbias=None, vbias=None, numpy_rng=None,
		theano_rng=None):
		"""
		RBM constructor.
		the parameters of RBM model are all None in default,
		waiting to initialize in the function.
		The rng is passed as parameter too, so that all rng 
		shared the same seed?
		"""

		# set attribute
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			# create a number generator
			numpy_rng = numpy.random.RandomState(1234)

		# what is the differences between numpy_rng and theano_rng?
		# numpy_rng returns a numpy object, which is a actual number or array
		# theano_rng returns a symbolic, which is still a TensorType with random property
		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if W is None:
			# W is initialized with common sample approach
			initial_W = numpy.asarray(numpy_rng.uniform(
					low = -4 * numpy.sqrt(6.0 / (n_hidden + n_visible)),
					high = 4 * numpy.sqrt(6.0 / (n_hidden + n_visible)),
					size = (n_visible, n_hidden)),
					dtype = theano.config.floatX
				)
			# theano shared variables for weights and biases
			W = theano.shared(value=initial_W, name='W', borrow=True)
		
		if hbias is None:
			hbias = theano.shared(value=numpy.zeros(n_hidden,
													dtype=theano.config.floatX),
									name='hbias', borrow=True)

		if vbias is None:
			vbias = theano.shared(value=numpy.zeros(n_visible,
													dtype=theano.config.floatX),
									name='vbias', borrow=True)

		# initialize input layer for standalone RBM or layer0 of DBN
		self.input = input
		if not input:
			self.input = T.matrix('input')

		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.theano_rng = theano_rng
		# store all parameters of model
		self.params = [self.W, self.hbias, self.vbias]

	def free_energy(self, v_sample):
		""" Function to compute the free energy """		
		wx_b = T.dot(v_sample, self.W) + self.hbias
		vbias_term = T.dot(v_sample, self.vbias)
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis = 1)	# free energy function with binary units.
		return -hidden_term - vbias_term

	def propup(self, vis):
		""" This function propagates the visible units activation upwards to the hidden units.

		Note that we return also the pre-sigmoid activation of the layer
		"""

		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias

		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_h_given_v(self, v0_sample):
		""" This function infers state of hidden units given visible units """

		# compute the activation of the hidden units
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)	# sigmoid's output is between 0~1, it's a nature probability (mean)?

		# get a sample of the hiddens given their activation
		# note that theano_rng.binomail returns a symbolic sample of dtype int 64 by default
		h1_sample = self.theano_rng.binomail(size=h1_mean.shape,
											n=1, p=h1_mean,
											dtype=theano.config.floatX)

		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def propdown(self, hid):
		""" This function propagates the hidden units activation downwards to the visible units """

		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias # the matrix W should transposition
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_v_given_h(self, h0_sample):
		""" This funtion infers state of visible units given hidden units"""
		pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
		# sample by rng
		v1_sample = self.theano_rng.binomail(size=v1_mean.shape, 
											n = 1, p=v1_mean,
											dtype=theano.config.floatX)
		return [pre_sigmoid_v1, v1_mean, v1_sample]

	def gibbs_hvh(self, h0_sample):
		""" This function implements one step of Gibbs sampling, starting from the hidden state"""
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
				pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample):
		""" This function implements one step of Gibbs sampling, starting from the visible state"""
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample,
				pre_sigmoid_v1, v1_mean, v1_sample]

	def get_cost_updates(self, lr=0.1, persistent=None, k=1):
		""" This functions implements one step of CD-k or PCD-k """
		# compute positive phase
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

		# decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample
		# for PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		# perform actual negative phase
		# in order to implement CD-k/PCD-k we need to scan over the function that implements one gibbs 
		# step k times. The scan will return the entire Gibbs chain
		[pre_sgimoid_nvs, nv_means, nv_samples,
		 pre_sigmoid_nhs, nh_means, nh_samples], updates = \
		 	theano.scan(self.gibbs_hvh,
		 			# the None are place holders, saying that chain_start is the initial state
		 			# corresponding to the 6th output)
					outputs_info=[None, None, None, None, None, chain_start],
					n_steps=k)

		# determine gradients on RBM parameters
		# note that we only need the sample at the end of the chain
		chain_end = nv_samples[-1]	# visible sample?

		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))

		# We must not compute the gradient through the gibbs sampling
		gparams = T.grad(cost, self.params, consider_constant=[chain_end])

		# constructs the update dictionary
		for gparam, param in zip(gparams, self.params):
			# make sure that the learning rate is of the right dtype
			updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

		if persistent:
			# Note that this works only if persistent is a shared varialble
			updates[persistent] = nh_samples[-1]
			# pseudo-likelihood is a better proxy for PCD
			# why not use the cost?
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		else:
			# reconstruction cross-entorpy is a better proxy for CD
			monitoring_cost = selft.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

		return monitoring_cost, updates 		# the variable 'updates' is filled by followed procedure, not by the scan

	def get_pseudo_likelihood_cost(self, updates):
		""" Stochastic approximation to the pseudo-likelihood 
			I have no idea why to do this.
		"""

		# index of bit i in expression p(x_i | x{\i})
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')

		# binarize the input image by rounding to nearest integer
		xi = T.round(self.input)	# input? It seems that the sample result has nothing to do with the cost...

		# calculate free energy for the given bit configuration
		fe_xi = self.free_energy(xi)

		# flip bit x_i of matrix xi and preserve all other bits x_{\i}
		xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

		# calculate free energy with bit flipped
		fe_xi_flip = self.free_energy(xi_flip)

		cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

		updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

		return cost

	def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
		""" Approximation to the reconstruction error 

		There are something about theano optimization
		"""
		cross_entropy = T.mean(
				T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
				(1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
					axis = 1)
			)

		return cross_entropy

def test_rbm(learning_rate = 0.1, training_epochs=15,
			 dataset='./data/mnist.pkl.gz', batch_size=20,
			 n_chains=20, nh_samples=10, output_folder='rbm_plots',
			 n_hidden=500):
	"""
	Demonstrate how to train and afterwards sample from it using Theano.

	This is demonstrated on MNIST.

	:param learning_rate: learning rate used for training the RBM

	:param training_epochs: number of epochs used for training

	:param dataset: path the the pickled dataset

	:param batch_size: size of a batch used to train the RBM

	:param n_chains: number of parallel Gibbs chains to be used for sampling

	:param n_samples: number of samples to plot for each chain

	"""
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]
	test_set_x, test_set_y = datasets[2]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

	# allocate symbolic variables for the data
	index = T.lscalar()
	x = T.matrix('x')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	# initialize storage for the persistent chain (state = hidden layer of chain)
	persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
												 dtype=theano.config.floatX),
									 borrow=True)

	# construct the RBM class
	rbm = RBM(input=x, n_visible=28 * 28,
			  n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

	# get the cost and the gradient corresponding to one step of CD-15
	cost, updates = rbm.get_cost_updates(lr=learning_rate,
										 persistent=persistent_chain, k=15)

	####################
	# Training the RBM #
	####################
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)	
	os.chdir(output_folder)


	# it is ok for a theano fnction to have no ouput
	# the purpose of train_rbm is solely to update the RBM parameters
	train_rbm = theano.function([index], cost,
			updates=updates,
			givens={x, train_set_x[index * batch_size:
									(index + 1) * batch_size]},
			name='train_rbm')

	plotting_time = 0.
	start_time = time.clock()

	# go through training epochs
	for epoch in xrange(training_epochs):

		# go through the training set
		mean_cost = []
		for batch_index in xrange(n_train_batches):
			mean_cost += [train_rbm(batch_index)]

		print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

		# Plot filters after each training epoch
		plotting_start = time.clock()
		# Construct image from the weight matrix
		image = PIL.Image.fromarray(title_raster_images(
					X=rbm.W.get_value(borrow=True).T,
					img_shape=(28, 28), tile_shape=(10, 10),
					tile_spacing=(1, 1)))
		image.save('filters_at_epoch_%i.png' % epoch)
		plotting_stop = time.clock()
		plotting_time += (plotting_stop - plotting_start)

	end_time = time.clock()

	pretraining_time = (end_time - start_time) - plotting_time

	print ('Training took %f minutes' % (pretraining_time / 60.))

	#########################
	# Sampling from the RBM #
	#########################
	# find out the number of test samples
	number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

	# pick random test examples, with which to initialize the persistent chain
	test_idx = rng.randint(number_of_test_samples - n_chains)	
	persistent_vis_chain = theano.shared(numpy.asarray(
			test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
			dtype=theano.config.floatX))
	
	plot_every = 1000
	# define one step of Gibbs sampling (mf = mean-field) define a 
	# function that does 'plot_every' steps before returning the
	# sample for plotting
	[presig_hids, hid_mfs, hid_samples, presig_vis,
	 vis_mfs, vis_samples], updates = \
	 					theano.scan(rbm.gibbs_vhv,
	 							outputs_infor[None, None, None, None,
	 										  None, persistent_vis_chain],
	 							n_steps=plot_every)	

	# add to updates the shared variable that takes care of out persistent
	# chain :.
	updates.update({persistent_vis_chain: vis_samples[-1]})

	# construct the function that implements out persistent chain.
	# we generate the "mean field" activations for plotting and the actual
	# samples for reinitializing the state of out persistent chain
	sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
								updates=updates,
								name='sample_fn')
	# create a space to store the image for plotting ( we need to leave
	# room for the tile_spacing as well)
	image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1),
							 dtype='uint8')
	for idx in xrange(n_samples):
		# generate `plot_every` intermediate samples that we discard,
		# because successive samples in the chain are too correlated
		vis_mf, vis_sample = sample_fn()
		print ' ... plotting sample ', idx
		image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
				X=vis_mf,
				img_shape=(28, 28),
				tile_shape=(1, n_chains),
				tile_spacing=(1, 1))
		# construct image

	image = PIL.Image.fromarray(image_data)
	image.save('samples.png')
	os.chdir('../')

if __name__ == '__main__':
	test_rbm()