import theano
import theano.tensor as T
import numpy
import cPickle, gzip, os, sys, time

print '... loading data'

# load the dataset
f = gzip.open('./data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# wrap to numpy array type
train_set_array = numpy.asarray(train_set[0], dtype='float32')
valid_set_array = numpy.asarray(valid_set[0], dtype='float32')
test_set_array = numpy.asarray(test_set[0], dtype='float32')

print "Shape of train_set is: " , train_set_array.shape
print "Shape of valid_set is: " , valid_set_array.shape
print "Shape of test_set is: " , test_set_array.shape

print "Shape of first mini-batch train_set is: " , train_set_array[0:600].shape
print "Shape of first mini-batch valid_set is: " , valid_set_array[0:600].shape
print "Shape of first mini-batch test_set is: " , test_set_array[0:600].shape