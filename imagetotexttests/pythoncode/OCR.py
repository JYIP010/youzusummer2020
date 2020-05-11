import pytesseract
from PIL import Image
import sys
from pdf2image import convert_from_path
import os
import io

pdf_path = "Sample Resources/P6_English_2019_CA1_CHIJ.pdf"
output_filename = "results.txt"
pages = convert_from_path(pdf_path)
pg_cntr = 1

sub_dir = str("images/" + pdf_path.split('/')[-1].replace('.pdf','') + "/")
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)

for page in pages:
    filename = "pg_"+str(pg_cntr)+'_'+pdf_path.split('/')[-1].replace('.pdf','.jpg')
    page.save(sub_dir+filename)
    with io.open(output_filename, 'a+', encoding='utf8') as f:
        f.write(str("======================================================== PAGE " + str(pg_cntr) + " ========================================================\n"))
        f.write(str(pytesseract.image_to_string(sub_dir+filename)+"\n"))
        f.write(str("======================================================== ========================= ========================================================\n"))
    pg_cntr = pg_cntr + 1
