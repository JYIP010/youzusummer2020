import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import math as m

def mergeBoxes(image, imgCont):
    def getXFromRect(item):
        return item[0]


    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    j = 1
    POI_detected = []
    POI_coordinates = []
    acceptedRects = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        area = ((x + w) * (y * h)) / 100000
        if (x + w) > 300 and h < 100 and area < 2000:
            if POI_detected.count((str(m.floor(x)), str(m.floor(y)))) == 0 and (x, y) != (0, 0):
                POI_detected.append((str(m.floor(x)), str(m.floor(y))))
                POI_coordinates.append((x, y, w, h))
                print("Length of box = x + w = " + str(x + w))
                print("Height of box = h = " + str(h))
                #cv2.rectangle(imgCont, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print("POI " + str(j) + ": (" + str(int(x)) + ", " + str(int(y)) + ")")
                j = j + 1

    # We begin merging the boxes that have passed the initial filter
    # Sort bounding rects by x coordinate
    POI_coordinates.sort(key=getXFromRect)
    # Array of accepted rects
    # Merge threshold for x coordinate distance
    xThr = 5 # IMPORTANT TO EDIT
    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(POI_coordinates):
        currxMin = supVal[0]
        currxMax = supVal[0] + supVal[2]
        curryMin = supVal[1]
        curryMax = supVal[1] + supVal[3]

        # Iterate all initial bounding rects
        # starting from the next
        for subIdx, subVal in enumerate(POI_coordinates[(supIdx + 1):], start=(supIdx + 1)):

            # Initialize merge candidate
            candxMin = subVal[0]
            candxMax = subVal[0] + subVal[2]
            candyMin = subVal[1]
            candyMax = subVal[1] + subVal[3]

            # Check if x distance between current rect
            # and merge candidate is small enough
            if (candxMin <= currxMax + xThr):
                # Reset coordinates of current rect
                currxMax = candxMax
                curryMin = min(curryMin, candyMin)
                curryMax = max(curryMax, candyMax)
                if acceptedRects.count([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin]) == 0:
                    acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
            else:
                if acceptedRects.count([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin]) == 0:
                    acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

    # No more merge candidates possible, accept current rect

    print("Coordinates of accepted rectangles: " + str(acceptedRects))
    print(len(acceptedRects))
    print(type(acceptedRects))
    for rectIndx, rect in enumerate(acceptedRects):
        x_ = rect[0]
        y_ = rect[1]
        w_ = rect[2]
        h_ = rect[3]
        if h_ < 50:
            cv2.rectangle(imgCont, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)

