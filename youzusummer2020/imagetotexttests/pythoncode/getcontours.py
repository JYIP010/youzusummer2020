import cv2
import numpy as np
import math as m


#frameWidth = 640
#frameHeight = 480
'''
#stack function
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
'''


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 1
    POI_detected = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x_, y_, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(cnt)
        if area > 600 and (x_ + w) > 10 and h < 100:
            if POI_detected.count((str(m.floor(x_)), str(m.floor(y_)))) == 0 and (x_, y_) != (0, 0):
                POI_detected.append((str(m.floor(x_)), str(m.floor(y_))))
                #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
                cv2.rectangle(imgContour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 3)
                #cv2.putText(imgContour, "Point " + str(i), (x_ + w + 5, y_ + 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
                #cv2.putText(imgContour, "Coordinates: (" + str(int(x_)) + ", " + str(int(y_)) + ")", (x_ + w + 20, y_ + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
                print("POI " + str(i) + ": (" +  str(int(x_)) + ", " + str(int(y_)) + ")")
                i = i + 1
    print("List of coordinates of points detected: " + str(POI_detected))


def merge_boxes(box1, box2):
    return [min(box1[0], box2[0]),
         min(box1[1], box2[1]),
         max(box1[2], box2[2]),
         max(box1[3], box2[3])]


def empty(a):
    pass

'''
#creating the trackbar
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 0, 0)
cv2.createTrackbar("Threshold1","Parameters", 255, 255, empty)
cv2.createTrackbar("Threshold2","Parameters", 23, 255, empty)
'''

#instantiating the image
#img = cv2.imread("/home/justinyip/Documents/youzu2020/youzu-master/Sample Resources/sciencetest.jpg")
img = cv2.imread("/home/justinyip/Documents/youzu2020/youzu-master/Sample Resources/engtest.jpg")
imgBlur = cv2.GaussianBlur(img, (7,7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

'''
imgS = cv2.resize(img, (600, 750))
imgBlurS = cv2.resize(imgBlur, (600, 750))
imgGrayS = cv2.resize(imgGray, (600, 750))
'''

'''
threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
'''
#imgCanny = cv2.Canny(imgGrayS, threshold1, threshold2)
imgCanny = cv2.Canny(imgGray, 255, 23)
kernel = np.ones((5,5))
imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

#imgContour = imgS.copy()
imgContour = img.copy()
getContours(imgDil, imgContour)
imgContouredS = cv2.resize(imgContour, (600, 750))
'''
imgstack = stackImages(0.8, ([imgS, imgGrayS, imgCanny], [imgDil, imgContour, imgContour]))
imgstackS = cv2.resize(imgstack, (800,800))
cv2.imshow("Result", imgstackS)
'''

cv2.imshow("Result", imgContouredS)
cv2.waitKey(0)
