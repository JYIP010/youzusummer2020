# V1: Basic implementation of contour detectiong
# Converts a negative image to positive image
# Sort the contours according to y coordinates

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return pixel_values

def is_white_image(image_name):
    numpy_array = get_image("Sample Resources/" + image_name + ".jpg")
    total_pixels = numpy_array.size
    num_of_white = np.count_nonzero(numpy_array == [255,255, 255])
    num_of_black = np.count_nonzero(numpy_array == [1,1, 1])
    white_percentage = num_of_white / total_pixels
    #Save as inverted image if it is a negative image
    if white_percentage < 0.75:
        # load image# Load image 
        im = Image.open("Sample Resources/" + image_name + ".jpg")

        # Invert
        result = ImageOps.invert(im)

        # Save
        result.save("Sample Resources/" + image_name + "_inverted.jpg")
        return False
    else:
        return True

image_name = "/table_17"

if not is_white_image(image_name):
    image_name = image_name + "_inverted"

img = cv2.imread("Sample Resources/" + image_name + ".jpg")
height, width, channels = img.shape
#Blurring
imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
# cv2.imshow("blur",imgBlur)

# convert to gray
gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",gray)
# threshold the grayscale image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Draw contours
result = img.copy()

# use morphology erode to blur horizontally
#kernel = np.ones((500,3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 3)) #250,3
morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17)) #3,17
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

resized=cv2.resize(morph,(700,850))
#cv2.imshow("morph",resized)

# find contours
cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

ordered_value_tuples = []
i=0
for c in cntrs:
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    # if h < 50 and w > 400 :
    if True:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 2)
        # print("Box: " + str(i) + ": (" + str(int(x)) + ", " + str(int(y)) + "," + str(int(w)) + "," + str(int(h)) + ")")
        # print('Area: ' + str(area))
        i += 1

        
        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

        image = cv2.imread("Sample Resources/" + image_name + ".jpg", 0)
        thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ROI = thresh[y:y+h,x:x+w]
        data = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
        
        ordered_value_tuples.append((data, y))
        ordered_value_tuples.sort(key = lambda tup: tup[1])

for value in ordered_value_tuples:
    print(value)

# write result to disk
#cv2.imwrite("test_text_threshold.png", thresh)
#cv2.imwrite("test_text_morph.png", morph)
#cv2.imwrite("test_text_lines.jpg", result)

#cv2.imshow("GRAY", gray)
#cv2.imshow("THRESH", thresh)
#cv2.imshow("MORPH", morph)

ims=cv2.resize(result,(700,850))
cv2.imshow("RESULT", ims)
cv2.waitKey(0)
cv2.destroyAllWindows()