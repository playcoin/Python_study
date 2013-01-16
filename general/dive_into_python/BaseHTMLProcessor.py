"Dive into Python example 8.1"
from sgmllib import SGMLParser
import htmlentitydefs

class BaseHTMLProcessor(SGMLParser):
	def reset(self):
		self.pieces = []
		SGMLParser.reset(self)

	def unknown_starttag(self, tag, attrs):
		strattrs = "".join([' %s="%s"' % (k,v) for k, v in attrs])
		self.pieces.append("<%(tag)s%(strattrs)s>" % locals())

	def unknown_endtag(self, ref):
		self.pieces.append("</%(tag)s>" % locals())

	def handle_charref(self, ref):
		# ref does not include "&#" and ";"
		self.pieces.append("&#%(ref)s;" % locals())

	def handle_entityref(self, ref):
		self.pieces.append("&%(ref)s" % locals())
		# only some entity in certain standard HTML need to append ';'
		if htmlentitydefs.entitydefs.has_key(ref):
			self.pieces.append(';')

	def handle_data(self, text):
		self.pieces.append(text)

	def handle_comment(self, text):
		self.pieces.append("<!-%(text)s->" % locals())

	def handle_pi(self, text):
		self.pieces.append("<?%(text)s" % locals())

	def handle_decl(self, text):
		self.pieces.append("<!%(text)s" % locals())

	def output(self):
		return "".join(self.pieces)

if __name__ == "__main__":
	for k,v in globals().items():
		print k, '=', v