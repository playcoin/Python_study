import urllib
import cPickle

data = urllib.urlopen("http://www.pythonchallenge.com/pc/def/banner.p")
load_obj = cPickle.load(data)
data.close()
text = ""
# use the iterator!
for row in load_obj:
	# use the iterator and tuple object!
	for ch, count in row:
		text += ch * count
	text += "\n"

text_file = file('5.text', 'wb')
text_file.write(text)
text_file.close()