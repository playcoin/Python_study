"""Framework for getting filetype-specific metadata"""

import os, sys
from UserDict import UserDict

def stripnulls(data):
	"strip whitespace and nulls"
	# \OO means what?
	return data.replace("\OO", "").strip()

class FileInfo(UserDict):	# extends from UserDict
	"store file metadata"
	def __init__(self, filename=None):
		UserDict.__init__(self)	# call the __init__ function of the super class explicitly!!
		self["name"] = filename # self is a instance of UserDict, so it could use [] operator!

class MP3FileInfo(FileInfo): # extends from FileInfo
	"store ID3V1.0 MP3 tags"
	tagDataMap = {
		"title"  : (3, 33, stripnulls),	# the start and end position, and the parseFunc
		"artist" : (33, 63, stripnulls),
		"album"	 : (63, 93, stripnulls),
		"year"	 : (93, 97, stripnulls),
		"comment"	 : (97, 126, stripnulls),
		"genre"	 : (127, 128, ord)
	}

	def __parse(self, filename):
		"parse ID3V1.0 tags from MP3 file"
		self.clear()
		try:
			fsock = open(filename, 'rb', 0)
			try:
				fsock.seek(-128, os.SEEK_END) # set seek position to the 128 to last
				tagdata = fsock.read(128)	# read the last 128 characters
			finally:
				fsock.close()								

			if tagdata[:3] == "TAG": # check ID3V1?
				for tag, (start, end, parseFunc) in self.tagDataMap.items():
					self[tag] = parseFunc(tagdata[start:end])
		except IOError:
			pass

	def __setitem__(self, key, item): # use the [] operator to set attribute
		# The code "self["name"] = filename" in the __init__ function of the super class FileInfo
		# will call this function ?
		if key == "name" and item:
			self.__parse(item)
		FileInfo.__setitem__(self, key, item)

def listDirectory(directory, fileExtList):
	"get list of file info objects for files of particular extensions"
	fileList = [os.path.normcase(f)
		for f in os.listdir(directory) 
	]
	fileList = [os.path.join(directory, f)
		for f in fileList
		if os.path.splitext(f)[1] in fileExtList
	]

	def getFileInfoClass(filename, module=sys.modules[FileInfo.__module__]):
		"get file info class from filename extension"
		subclass = "%sFileInfo" % os.path.splitext(filename)[1].upper()[1:]
		return hasattr(module, subclass) and getattr(module, subclass) or FileInfo

	return [getFileInfoClass(f)(f) for f in fileList]

if __name__ == "__main__":
	for info in listDirectory("M:\\D\\music\\English", [".mp3"]):
		print "\n".join(["%s=%s" % (k, v) for k, v in info.items()])
		print