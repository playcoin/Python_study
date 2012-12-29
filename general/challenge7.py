import re
from PIL import Image

im = Image.open('oxygen.png')
print im.size

# get the grey box
im_crpy = im.crop((0, 43, 607, 44))

##############
#  METHOD 1  #
##############
## get the text from box, and remove all '\xff'
# text_l = im_crpy.tostring().replace('\xff', '')
# print text_l
# remove all the duplicate characters
# text = text_l[0]

# cur_ch = text_l[0]
# count = 0
# for ch in text_l:
# 	# add count in case the two same characters are adjacent
# 	if ch == cur_ch and count < 20:
# 		count += 1
# 		continue
# 	else:
# 		cur_ch = ch
# 		text += ch
# 		count = 0;

##############
#  METHOD 2  #
##############
text = ""
text_l = im_crpy.tostring()
for i in range(0, len(text_l), 28): # Why 28? Because each pixel is a 4-tuple, and each color box has 7 pixels.
	text += text_l[i]



print text

# get the array
matches = re.findall(r'\[[^\]]+\]', text)
# use 'eval' function
array = eval(matches[0])

print array

# how to decode the array?
decode = ""
for i in array:
	decode += chr(i)

print decode

##############
#  METHOD 3  #
##############
# cal directly, no need to get the subsection of the origin image
print im.tostring()[108188:110620:28]

