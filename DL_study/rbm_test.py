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

		
