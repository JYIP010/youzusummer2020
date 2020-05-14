import re
matches = re.match( r'[[({|IL][1-9]*[0-9]?[])}|]', "ewf [1]", re.I)

print(matches)