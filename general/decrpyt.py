# dicMap = {}
# dicMap['a'] = 'c'
# dicMap['b'] = 'd'
# dicMap['c'] = 'e'
# dicMap['d'] = 'f'
# dicMap['e'] = 'g'
# dicMap['f'] = 'h'
# dicMap['g'] = 'i'
# dicMap['h'] = 'j'
# dicMap['i'] = 'k'
# dicMap['j'] = 'l'
# dicMap['k'] = 'm'
# dicMap['l'] = 'n'
# dicMap['m'] = 'o'
# dicMap['n'] = 'p'
# dicMap['o'] = 'q'
# dicMap['p'] = 'r'
# dicMap['q'] = 's'
# dicMap['r'] = 't'
# dicMap['s'] = 'u'
# dicMap['t'] = 'v'
# dicMap['u'] = 'w'
# dicMap['v'] = 'x'
# dicMap['w'] = 'y'
# dicMap['x'] = 'z'
# dicMap['y'] = 'a'
# dicMap['z'] = 'b'
# dicMap[' '] = ' '
# dicMap['.'] = '.'
# dicMap["'"] = "'"
# dicMap["("] = "("
# dicMap[")"] = ")"

text = "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."

# for i in range(len(text)):
# 	print dicMap[text[i]],

import string 

print string.translate(text, string.maketrans("abcdefghijklmnopqrstuvwxyz", "cdefghijklmnopqrstuvwxyzab"))

