# zipfile module
import re
import zipfile

stop = False
z = zipfile.ZipFile('./channel.zip')
file_path = "90052.txt"
comments_all = ""
while stop is False:
	# read the API document. get the ZipInfo object
	file_info = z.getinfo(file_path)
	# the zipfile.read() can use filepath(str) or ZipInfo as parameter
	# get the bytes in the file
	file_read = z.read(file_info)

	next_n = re.findall(r'\d+', file_read)
	
	# print next_n
	if len(next_n) < 1:
		stop = True
	else:
		file_path = "%s.txt" % next_n[0]

	# the the ZipInfo can have comment of their each own.
	comments_all += file_info.comment


z.close()

print "Over! The last url is: " , file_path
print "Comments is :\n" , comments_all