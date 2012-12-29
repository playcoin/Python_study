import re
from PIL import Image

im = Image.open('oxygen.png')
print im.size

# get the grey box
im_crpy = im.crop((0, 43, 607, 51))

# get the text from box, and remove all '\xff'
text_l = im_crpy.tostring().replace('\xff', '')

# remove all the duplicate characters
text = text_l[0]

cur_ch = text_l[0]
for ch in text_l:
	if ch == cur_ch:
		continue
	else:
		cur_ch = ch
		text += ch

print text

matches = re.findall(r'\[[^\]]+\]', text)

array = eval(matches[0])

print array

decode = ""
for i in array:
	decode += text[i]

print decode