#conding:utf-8
import pdb
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

from detect import *
from svm_train import *

SZ = 20          # size of training images
MAX_WIDTH = 1000 # maximum width of original image
MIN_AREA = 2000  # minimum area of license plate
PROVINCE_START = 1000

svm_model = SVM(C=1, gamma=0.5)
model_1,model_2 = svm_model.train_svm()



def pfz(msg):
    print(msg)

def find_waves(threshold, histogram):
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks



def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards

def Cardseg(rois, colors):
    seg_dic = {}
    old_seg_dic = {}
    for i, color in enumerate(colors):
        if color in ("blue", "yello", "green"):
            card_img = rois[i]
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            x_histogram  = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram)/x_histogram.shape[0]
            x_threshold = (x_min + x_average)/2
            wave_peaks = find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                continue

            wave = max(wave_peaks, key=lambda x:x[1]-x[0])
            gray_img = gray_img[wave[0]:wave[1]]

            row_num, col_num= gray_img.shape[:2]
            gray_img = gray_img[1:row_num-1]
            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram)/y_histogram.shape[0]
            y_threshold = (y_min + y_average)/5

            wave_peaks = find_waves(y_threshold, y_histogram)

            if len(wave_peaks) <= 6:
                continue
            wave = max(wave_peaks, key=lambda x:x[1]-x[0])
            max_wave_dis = wave[1] - wave[0]
            
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)
            
            cur_dis = 0
            for i,wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i+1:]
                wave_peaks.insert(0, wave)
            
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis/3:
                point_img = gray_img[:,point[0]:point[1]]
                if np.mean(point_img) < 255/5:
                    wave_peaks.pop(2)
            
            if len(wave_peaks) <= 6:
                continue
            pdb.set_trace()
            part_cards = seperate_card(gray_img, wave_peaks)
             
            predict_result = []
            for i, part_card in enumerate(part_cards):
                if np.mean(part_card) < 255/5:
                    continue
                part_card_old = part_card
                w = abs(part_card.shape[1] - SZ)//2
                
                part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
                part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                
                # hog feature 
                part_card = preprocess_hog([part_card])
                if i == 0:
                    resp = model_2.predict(part_card)
                    charactor = provinces[int(resp[0]) - PROVINCE_START]
                else:
                    resp = model_1.predict(part_card)
                    charactor = chr(resp[0])
                if charactor == "1" and i == len(part_cards)-1:
                    if part_card_old.shape[0]/part_card_old.shape[1] >= 7:
                        continue
                predict_result.append(charactor)

            seg_dic[i] = part_cards
            old_seg_dic[i] = part_card_old
    return seg_dic, old_seg_dic, predict_result


if __name__ == "__main__":

    for pic_file in os.listdir("./test_img"):
        
        img_path = os.path.join("./test_img", pic_file)
        pfz("img_path:%s"%(img_path))
        pfz("Start CaridDetect")
        roi, label, color = CaridDetect(img_path)
        pfz("End CaridDetect")

        pfz("Start Carseg")
        seg_dict, _ , pre= Cardseg([roi], [color])
        pfz("End Carseg")
    
        img = cv2.imread(img_path)
        img = cv2.rectangle(img, 
	(label[0], label[2]), (label[1], label[3]), (255, 0, 0), 2)
        
        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)))
        print(pre)
        cv2.imshow('show', img)
        pfz("imshow")
        
        cv2.waitKey(20000)
