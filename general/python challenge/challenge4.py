## Do some network job
import urllib
import re

stop = False
remote_url = "http://www.pythonchallenge.com/pc/def/linkedlist.php?nothing=63579"
while not stop:
	# open the url, get the network object
	data = urllib.urlopen(remote_url)
	print remote_url
	text = data.readline()
	print text
	next_n = re.findall(r'\d+', text)
	print next_n
	if len(next_n) < 1:
		stop = True
	else:
		remote_url = "http://www.pythonchallenge.com/pc/def/linkedlist.php?nothing=%s" % next_n[0]

	data.close()

print "Over! The last url is: " , remote_url