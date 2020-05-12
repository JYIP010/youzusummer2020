
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os

'''
# 图像切割模块
class cutImage(object):
    def __init__(self,img, bin_threshold, kernel, iterations, areaRange, filename, border=10, show=True, write=True,):

        :param img: 输入图像
        :param bin_threshold: 二值化的阈值大小
        :param kernel: 形态学kernel
        :param iterations: 迭代次数
        :param areaRange: 面积范围
        :param filename:保留json数据的文件名称
        :param border: 留边大小
        :param show: 是否显示结果图，默认是显示
        :param write: 是否把结果写到文件，默认是写入
    
        self.img = img
        self.bin_threshold = bin_threshold
        self.kernel = kernel
        self.iterations = iterations
        self.areaRange = areaRange
        self.border = border
        self.show = show
        self.write = write
        self.filename = filename
        
'''
# 检测表格，使用形态学
# 返回是表格图以及表格中交叉点的图
class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:  # 灰度图
            gray_img = self.src_img
        elif len(self.src_img.shape) ==3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1]/scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1)) # 形态学因子
        h_erode_img = cv2.erode(h_img,h_structure,1)

        h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)
        # cv2.imshow("h_erode",h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        mask_img = h_dilate_img+v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
        cv2.imshow("joints",joints_img)
        cv2.imshow("mask",mask_img)

        imgBlur = cv2.GaussianBlur(mask_img, (7, 7), 1)
        # cv2.imshow("blur",imgBlur)

        # convert to gray
        gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray",gray)

        # threshold the grayscale image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Draw contours
        result = self.src_img.copy()

        # use morphology erode to blur horizontally
        # kernel = np.ones((500,3), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 3))  # 250,3
        morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))  # 3,17
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        return mask_img,joints_img




def findboxes(result, img, cntrs, image_name):
    for c in cntrs:
        area = cv2.contourArea(c) / 10000
        x, y, w, h = cv2.boundingRect(c)
        # if h < 50 and w > 400 :
        if True:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # print("Box: " + str(i) + ": (" + str(int(x)) + ", " + str(int(y)) + "," + str(int(w)) + "," + str(int(h)) + ")")
            # print('Area: ' + str(area))
            i += 1
            # pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

            image = cv2.imread("Sample Resources/" + image_name + ".jpg", 0)
            thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            ROI = thresh[y:y + h, x:x + w]
            text = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')

            # Only add the image if it is large enough,
            # and the entire image has illegal text symbols(which is likely to be a diagram)
            if w > 50 and h > 50:
                if is_gibberish(text):
                    new_image = img[y:y + h, x:x + w]
                    cv2.imwrite("TempImages/" + str(count) + ".jpg", new_image)
                    document_data_list.append(("TempImages/" + str(count) + ".jpg", "image", y))
                    count = count + 1

            ordered_value_tuples.append((text, y))
            ordered_value_tuples.sort(key=lambda tup: tup[1])


if __name__=='__main__':
    img = cv2.imread('TempImages/1.jpg')
    #imgS = cv2.resize(img, (600, 750))
    cv2.imshow("img", img)
    mask, joint = detectTable(img).run()
    cv2.waitKey()