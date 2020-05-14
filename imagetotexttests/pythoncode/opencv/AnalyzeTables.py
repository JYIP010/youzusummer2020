#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os
import pytesseract
import json
import pandas as pd
from PIL import Image, ImageOps
from ExtractText import get_text

class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
        elif len(self.src_img.shape) ==3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1]/scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1))
        h_erode_img = cv2.erode(h_img,h_structure,1)

        h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size)) 
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        mask_img = h_dilate_img+v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
        cv2.imshow("joints",joints_img)
        cv2.imshow("mask",mask_img)
        cntrs = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS    )
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        ordered_value_tuples = []

        first_count = 0
        largest_area = -1
        count_to_ignore = -1
        for c in cntrs:
            # Loop through contours to find the largest contour to be ignored(which is likely to be the entire table)
            area = cv2.contourArea(c)/10000
            if area > largest_area:
                largest_area = area
                count_to_ignore = first_count
            first_count = first_count + 1

        second_count = 0
        for c in cntrs:
            area = cv2.contourArea(c)/10000
            total_height, total_width, total_channels = self.src_img.shape
            relative_size = area / total_height * total_width
            # ignore images that take up the entire image
            

            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(self.src_img, (x, y), (x+w, y+h), (255, 255, 255), 7)
            pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

            ROI = self.src_img[y:y+h,x:x+w]
            text = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
            cv2.imwrite("TempImages/" + "table_" + str(second_count) + ".jpg" , ROI)
            cell_value = get_text("table_" + str(second_count))
            if second_count != count_to_ignore:
                ordered_value_tuples.append((cell_value, x, y))
                ordered_value_tuples.sort(key = lambda tup: tup[2])
            second_count = second_count + 1

        print(ordered_value_tuples)
        table = sort_table(ordered_value_tuples)

        return mask_img,joints_img


def sort_table(ordered_value_tuples):
    temp_table = {}
    cur_y_value = -1
    cur_ordered_row = []
    row_num = 1

    for cell in ordered_value_tuples:
        value = cell[0]
        x = cell[1]
        y = cell[2]
        if cur_y_value == -1 or abs(cur_y_value - y) < 5:
            # If y value is within 5 pixels, they are considered to be in the same row
            cur_ordered_row.append((value, x))
            cur_y_value = y
        else:
            # Sort each row by the x value
            cur_ordered_row.sort(key = lambda tup: tup[1])
            temp_table[row_num] = cur_ordered_row
            row_num = row_num + 1

            cur_ordered_row = []
            cur_ordered_row.append((value, x))
            cur_y_value = y

    cur_ordered_row.sort(key = lambda tup: tup[1])
    temp_table[row_num] = cur_ordered_row

    output_table = {}
    row_count = 1
    max_row_length = 0
    for key, value in temp_table.items():
        row_without_empty_values = []
        for cell in value:
            if cell[0] != "":
                row_without_empty_values.append(cell[0])

        if len(row_without_empty_values) > max_row_length:
            max_row_length = len(row_without_empty_values)

        is_valid_table = False
        for cell in row_without_empty_values:
            if cell != "":
                is_valid_table = True
                break
        if is_valid_table:
            output_table[row_count] = row_without_empty_values
            row_count = row_count + 1

    cols = list(range(1, max_row_length + 1))
    df = pd.DataFrame(columns = cols)
    # Saves table to csv
    for key, value in output_table.items():
        if len(value) != max_row_length:
            # Inconsistent row lengths in the table
            diff = max_row_length - len(value)
            for i in range(diff):
                value.append("")

        df.loc[key] = value


    output_table_json = json.dumps(output_table)
    df.to_csv("table.csv")
    print(output_table_json)
    return output_table_json




if __name__=='__main__':
    img = cv2.imread('TempImages/17.jpg')
    cv2.imshow("img",img)
    mask,joint = detectTable(img).run()
    cv2.waitKey()
    


