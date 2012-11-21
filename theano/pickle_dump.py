import cPickle

# open file
f = file('obj.save', 'wb')

# initial a variable
my_obj = [1, 2, 3]

cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f1 = file('dump/objs.save', 'wb')
for obj in [1, 2, 3, 4] :
	cPickle.dump(obj, f1, protocol=cPickle.HIGHEST_PROTOCOL)

f1.close()