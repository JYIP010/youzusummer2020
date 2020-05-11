# this python module focuses on pytesseract to detect the text in the image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import math as m
import skew as sk
import boxmerger as bm
import boxmergeralt as bma

# load image
img = "/home/justinyip/Documents/youzu2020/youzu-master/Sample Resources/engtest.jpg"
#img = cv2.imread("/home/justinyip/Documents/youzu2020/youzu-master/Sample Resources/engtest.jpg")
img = sk.skew(img)

def getContours(image, imgContour):
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    j = 1
    POI_detected = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        area = ((x + w) * (y * h))/100000
        if (x + w) > 300 and h < 100 and area < 4000:
            if POI_detected.count((str(m.floor(x)), str(m.floor(y)))) == 0 and (x,y) != (0,0):
                POI_detected.append((str(m.floor(x)), str(m.floor(y))))
                print("Length of box = x + w = " + str(x + w))
                print("Height of box = h = " + str(h))
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.rectangle(imgContour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 3)
                #cv2.putText(imgContour, "Point " + str(j), (x + w + 5, y + 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
                #cv2.putText(imgContour, "Coordinates: (" + str(int(x)) + ", " + str(int(y)) + ")", (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
                print("POI " + str(j) + ": (" + str(int(x)) + ", " + str(int(y)) + ")")
                j = j + 1
    print("List of coordinates of points detected: " + str(POI_detected))


# image preprocessing
imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray, 23, 600)
kernel = np.ones((5, 5))
imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel, iterations=2)
imgOpen = cv2.morphologyEx(imgClose, cv2.MORPH_OPEN, kernel)
#imgDil = cv2.dilate(imgClose, kernel, iterations=1) RESULT: POORER RESOLUTION
#imgTophat = cv2.morphologyEx(imgOpen, cv2.MORPH_TOPHAT, kernel) RESULT: BLACK PAGE (SAME WITH BLACKHAT)

'''
d = pytesseract.image_to_data(imgDil, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(imgDil, (x, y), (x + w, y + h), (0, 255, 0), 2)
    box_coordinate = (x, y)
    print(box_coordinate)
'''
# resize to check each stage of preprocessing outcomes
#imgContour = img.copy()
imgContour1 = img.copy()
getContours(imgOpen, imgContour1)
imgContoured1S = cv2.resize(imgContour1, (600, 750))
#bm.mergeBoxes(imgOpen, imgContour)
#bma.mergeBoxes(imgOpen, imgContour)
imgContouredS = cv2.resize(imgContour1, (600, 750))
#imgMergedS = cv2.resize(imgContour, (600, 750))
#imgCloseS = cv2.resize(imgClose, (600, 750))
#imgOpenS = cv2.resize(imgOpen, (600, 750)) #only if need to check output before bounding boxes
#imgCannyS = cv2.resize(imgCanny, (600, 750))
#imgS = cv2.resize(img, (700, 850))
#cv2.imshow('img', imgMergedS)
cv2.imshow('withoutmerge', imgContoured1S)
#cv2.imshow('afterskew', imgS)
#cv2.imshow('test', imgOpenS)
#cv2.imshow('Cannytest', imgCannyS)

cv2.waitKey(0)