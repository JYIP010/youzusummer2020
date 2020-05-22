import glob
import win32com.client
import os

word = win32com.client.Dispatch("Word.Application")
word.visible = 0

doc = "science.pdf"
filename = doc.split('\\')[-1]
in_file = os.path.abspath(doc)
print(in_file)
wb = word.Documents.Open(in_file)
out_file = "science.docx"
print("outfile\n",out_file)
wb.SaveAs2(out_file, FileFormat=16) # file format for docx
print("success...")
wb.Close()

word.Quit()