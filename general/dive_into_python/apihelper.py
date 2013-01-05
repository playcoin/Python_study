# A API __doc__ list

def info(object, spacing=10, collapse=1):
	""" Read each callable methods' __doc__ of a given object
	"""

	# read the method list
	methodList = [method for method in dir(object) if callable(getattr(object, method))]
	# print function
	printfunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
	# print all __doc__
	print "\n".join(["%s %s" %
			(method.ljust(spacing),
			printfunc(str(getattr(object, method).__doc__)))
			for method in methodList
		  ])

if __name__ == "__main__":
	print info.__doc__