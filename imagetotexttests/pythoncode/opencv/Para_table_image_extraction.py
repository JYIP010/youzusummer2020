from docx2python import docx2python

# extract docx content, write images to image_directory
result = docx2python('science.docx', 'TempImages')
result_list = result.body
output_list = []

for i in result_list:
    for j in i:
        for k in j:
            for l in k:
                output_list.append(l)


with open('science.txt', 'w') as f:
    for item in output_list:
        f.write("%s\n" % item)