import re
mystr = "wdwdsvd]erg"
matches = re.search(r'[\[\(\|\{]+[1-9]+[o0-9]*[mn]*[\]\)\}\|]+', mystr, re.I)

if matches:
    print(True)
else:
    print(False)