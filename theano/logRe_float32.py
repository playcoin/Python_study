import numpy
import theano
import theano.tensor as T
rng = numpy.random

#assign floatX to float32
theano.config.floatX = 'float32'

N = 400
feats = 784
# in python the '(' can be seen as '[', when the elements in '()' splited by ',' ?
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 100

# Declare Theano symbolic variables
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "Initial modal:"
print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1        = 1 / (1 + T.exp(-T.dot(x, w) - b)) # Probability that target = 1
prediction = p_1 > 0.5					# The prediction thresholded
xent       = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1) # Cross-entropy loss function
cost       = xent.mean() + 0.01 * (w ** 2).sum()	#The cost to minimize
gw, gb     = T.grad(cost, [w, b]) 			# compute the gradient of the cost

# Compile
train = theano.function(
	inputs  = [x, y],
	outputs = [prediction, xent],
	updates = {w: w- 0.1 * gw, b: b - 0.1 * gb}
)
predict = theano.function(inputs=[x], outputs=prediction)

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        train.maker.fgraph.toposort()]):
    print 'Used the cpu'
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
          train.maker.fgraph.toposort()]):
    print 'Used the gpu'
else:
    print 'ERROR, not able to tell if theano used the cpu or the gpu'
    print train.maker.fgraph.toposort()

# train
for i in range(training_steps):
	pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])