#coding:utf-8
'''
表格生成线条坐标
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os
import pytesseract

# 图像切割模块
class cutImage(object):
    def __init__(self,img, bin_threshold, kernel, iterations, areaRange, filename, border=10, show=True, write=True,):
        '''
        :param img: 输入图像
        :param bin_threshold: 二值化的阈值大小
        :param kernel: 形态学kernel
        :param iterations: 迭代次数
        :param areaRange: 面积范围
        :param filename:保留json数据的文件名称
        :param border: 留边大小
        :param show: 是否显示结果图，默认是显示
        :param write: 是否把结果写到文件，默认是写入
        '''
        self.img = img
        self.bin_threshold = bin_threshold
        self.kernel = kernel
        self.iterations = iterations
        self.areaRange = areaRange
        self.border = border
        self.show = show
        self.write = write
        self.filename = filename
    def getRes(self):
        fl = open(self.filename,'w')
        if self.img.shape[2] == 1:  # 灰度图
            img_gray = self.img
        elif self.img.shape[2] ==3:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray,self.bin_threshold,255,cv2.THRESH_BINARY_INV) # 二值化
        img_erode = cv2.dilate(thresh, self.kernel, iterations=self.iterations)

        cv2.imshow('thresh',thresh)
        cv2.imshow('erode',img_erode)
        image, contours, hierarchy = cv2.findContours(img_erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        roiList = []
        res =[]
        result = {}
        area_coord_roi = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area >self.areaRange[0] and area <self.areaRange[1]:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = self.img[y+self.border:(y+h)-self.border,x+self.border:(x+w)-self.border]
                area_coord_roi.append((area,(x,y,w,h),roi))
        max_area = max([info[0] for info in area_coord_roi])

        for info in area_coord_roi:
            if info[0]==max_area:
                max_rect = info[1]
        for each in area_coord_roi:
            x,y,w,h = each[1]
            if x>max_rect[0] and y>max_rect[1] and (x+w)<(max_rect[0]+max_rect[2]) and (y+h) <(max_rect[1]+max_rect[3]):
                pass
            else:
                tmp_= each[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                tmp = []
                name = "tmp.jpg"
                cv2.imwrite(name,each[2])
                # text = image_to_string(name,False,'-l chi_sim')
                # tmp.append(text)
                tmp.append(" ")
                tmp.extend(list(tmp_))
                tmp.append("0 0 0")
                res.append(tmp)
                os.remove(name)
        cv2.imshow("yyy",img)

        result['1']=[res]
        fl.write(json.dumps(result))
        return roiList

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
        cntrs = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS    )
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        count = 0
        ordered_value_tuples = []
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
            cv2.imwrite("TempImages/" + "table_" + str(count) + ".jpg" , ROI)
            cell_value = get_text("table_" + str(count))
            print(cell_value)
            if True:
            #if cell_value != "": 
                if count != 0:
                    ordered_value_tuples.append((cell_value, x, y))
                    ordered_value_tuples.sort(key = lambda tup: tup[2])
                count = count + 1

        table = sort_table(ordered_value_tuples)
        for row in table:
            print(row)

        return mask_img,joints_img

            # 将生成的json数据显示在图像上
def drawLine(all_lines,height=841,width=595):
    blank_image = np.zeros((height, width, 3), np.int8)
    color = tuple(reversed((0,0,0)))
    blank_image[:] = color
    for _line in all_lines:
        for line in _line:
            if line[1]<0:
                line[1]=0
            if line[2]<0:
                line[2]=0
            if line[3]<0:
                line[3]=0
            if line[4]<0:
                line[4]=0

            p1=[int(np.round(line[1])),int(np.round(line[2]))]
            p2 = [int(np.round(line[1])+np.round(line[3])),int(np.round(line[2]))]
            p3 = [int(np.round(line[1])),int(np.round(line[2])+np.round(line[4]))]
            p4 = [int(np.round(line[1])+np.round(line[3])),int(np.round(line[2])+np.round(line[4]))]
            cv2.line(blank_image, (p1[0],p1[1]), (p2[0],p2[1]),(255, 0, 0),1)
            cv2.line(blank_image, (p1[0],p1[1]),(p3[0],p3[1]),(255, 0, 0),1)
            cv2.line(blank_image, (p2[0],p2[1]),(p4[0],p4[1]),(255, 0, 0),1)
            cv2.line(blank_image, (p3[0],p4[1]),(p4[0],p4[1]),(255, 0, 0),1)

    cv2.imshow("img",blank_image)
    cv2.waitKey()

def get_text(image_name):
    img = cv2.imread("TempImages/" + image_name + ".jpg")
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
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 2)

        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

        image = cv2.imread("TempImages/" + image_name + ".jpg", 0)
        thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ROI = thresh[y:y+h,x:x+w]
        data = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
        ordered_value_tuples.append((data, y))
        ordered_value_tuples.sort(key = lambda tup: tup[1])
        i = i + 1

    ordered_value_tuples = [i[0] for i in ordered_value_tuples]

    output_text = ""
    for value in ordered_value_tuples:
        output_text = output_text + value

    return output_text

def sort_table(ordered_value_tuples):
    temp_table = []
    cur_y_value = -1
    cur_ordered_row = []
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
            temp_table.append(cur_ordered_row)

            cur_ordered_row = []
            cur_ordered_row.append((value, x))
            cur_y_value = y

    cur_ordered_row.sort(key = lambda tup: tup[1])
    temp_table.append(cur_ordered_row)
    

    output_table = []
    for ordered_row in temp_table:
        new_ordered_row= [i[0] for i in ordered_row]
        output_table.append(new_ordered_row)

    return output_table




if __name__=='__main__':
    img = cv2.imread('1.jpg')
    cv2.imshow("img",img)
    mask,joint = detectTable(img).run()
    cv2.waitKey()


