import cPickle

f = file('dump/obj.save', 'rb')
loaded_obj = cPickle.load(f)

f.close()

print loaded_obj

# load sevral objects

f1 = file('dump/objs.save', 'rb')
loaded_objs = []
for i in range(4) :
	loaded_objs.append(cPickle.load(f1))

f.close()

print loaded_objs

# There some stuffs about Short or Long Term Serialization
# __getstate__ and __setstate__ 
# but i don't know what it means