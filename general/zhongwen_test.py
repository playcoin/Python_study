# -*- coding: utf-8 -*-
print "Directly: 中文"

a = {
	"中文" : 1,
	"英文" : 2
}

print a

print "{" ,
for b in a:
	print '"' + b + '" : ' + str(a[b]) + "," ,

print "}"


c = u"中文"

print c