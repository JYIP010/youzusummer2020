import cv2
import numpy as np
import pytesseract

# load image
img = cv2.imread("Sample Resources/pg_6_P6_Science_2019_SA2_CHIJ.jpg")

#deskew image
def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 +angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    correct = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return correct

a=deskew(img)

#Blurring
imgBlur = cv2.GaussianBlur(a, (7, 7), 1)
#cv2.imshow("blur",imgBlur)

# convert to gray
gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",gray)

# threshold the grayscale image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#cv2.imshow("thresh", thresh)

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

# Draw contours
result = img.copy()

i=0
for c in cntrs:
    area = cv2.contourArea(c)/10000
    x,y,w,h = cv2.boundingRect(c)
    if h < 150 and w > 400:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("Box: " + str(i) + ": (" + str(int(x)) + ", " + str(int(y)) + "," + str(int(w)) + "," + str(int(h)) + ")")
        print('Area: ' + str(area))
        i += 1
print(i)

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