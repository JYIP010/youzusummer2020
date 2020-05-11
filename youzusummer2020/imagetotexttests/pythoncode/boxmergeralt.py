import cv2
import opencv_wrapper as cvw
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

    sorted_rects = sorted(POI_coordinates, key=getXFromRect)

    # Distance threshold
    dt = 5

    # List of final, joined rectangles
    final_rects = [sorted_rects[0]]

    for rect in sorted_rects[1:]:
        prev_rect = final_rects[-1]

        # Shift rectangle `dt` back, to find out if they overlap
        shifted_rect = cvw.Rect(rect[0] - dt, rect[1], rect[2], rect[3])
        intersection = cvw.rect_intersection(prev_rect, shifted_rect)
        if intersection is not None:
            # Join the two rectangles
            min_y = min((prev_rect[1], rect[1]))
            max_y = max((prev_rect[1] + prev_rect[3], rect[1] + rect[3]))
            max_x = max((prev_rect[0] + prev_rect[2], rect[0] + rect[2]))
            width = max_x - prev_rect[0]
            height = max_y - min_y
            new_rect = cvw.Rect(prev_rect[0], min_y, width, height)
            # Add new rectangle to final list, making it the new prev_rect
            # in the next iteration
            final_rects[-1] = new_rect
        else:
            # If no intersection, add the box
            final_rects.append(rect)

    for rect in final_rects:
        cvw.rectangle(imgCont, rect, cvw.Color.GREEN, thickness=2)