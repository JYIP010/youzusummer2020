import re
import platform

sentence = "24 According to the webpage, which statement about Shakesphere Secondary School is |\ncorrect?"

search_sentence = re.match(r'[\[\(\|\{il]?[i1-9]*[o0-9]?[mn]?[\]\)\}\|]$|[\[\(\|\{il][i1-9]*[o0-9]?[mn]?[\]\)\}\|]?$', sentence, re.I)

if search_sentence:
    print("Matches")

else:
    print("Does not match")

if platform.system() == "Linux":
    print("It's Linux")