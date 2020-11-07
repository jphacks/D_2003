from __future__ import division
# -*- coding: utf-8 -*-
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl

import requests
from requests.auth import HTTPDigestAuth

import io
from PIL import Image, ImageDraw, ImageFilter

import play

import csv
import itertools
import math

import os
import shutil

from distutils.dir_util import copy_tree

import ifttt_demo2
import ifttt_demo_log
import play

count3=0




def moverecursively(source_folder, destination_folder):
    basename = os.path.basename(source_folder)
    dest_dir = os.path.join(destination_folder, basename)
    if not os.path.exists(dest_dir):
        shutil.move(source_folder, destination_folder)
    else:
        dst_path = os.path.join(destination_folder, basename)
        for root, dirs, files in os.walk(source_folder):
            for item in files:
                src_path = os.path.join(root, item)
                dst_file = os.path.join(root, item)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.move(src_path, dst_path)
            for item in dirs:
                src_path = os.path.join(root, item)
                moverecursively(src_path, dst_path)

def prep_image(img, inp_dim):
    # CNNに通すために画像を加工する
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    return img_, orig_im, dim

def count(x, img, count):
    # 画像に結果を描画
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    print("label:\n", label)
    # 人数カウント
    if(label=='no-mask'):
        count+=1
    print(count)

    return count

def write(x, img,camId):
    global count
    global point
    p = [0,0]
    # 画像に結果を描画
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    #print("x:", x)
    #print("cls:", cls)
    #print("len_classes:", len(classes))
    print(camId, "_c0:",c1)
    print(camId, "_c1:",c2)
    if cls != 0:
        cls = 1
    label = "{0}".format(classes[cls])
    print("label:", label)
    # 人数カウント
    if(label=='no-mask'):
        count+=1
    print(count)

    p[0] = (c2[0]+c1[0])/2
    p[1] = (c2[1]+c1[1])/2
    p[0] = p[0].cpu().numpy().copy()
    p[1] = p[1].cpu().numpy().copy()
    # p[0] = p[0].to('cpu').detach().numpy().copy()
    # p[1] = p[1].to('cpu').detach().numpy().copy()
    # p = p.astype(np.int64).tolist()
    point[camId].append(p)


    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

    return img


def arg_parse():
    # モジュールの引数を作成
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo') # ArgumentParserで引数を設定する
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    # confidenceは信頼性
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    # nms_threshは閾値

    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
                        # resoはCNNの入力解像度で、増加させると精度が上がるが、速度が低下する。
    return parser.parse_args() # 引数を解析し、返す

def cvpaste(img, imgback, x, y, angle, scale):
    # x and y are the distance from the center of the background image

    r = img.shape[0]
    c = img.shape[1]
    rb = imgback.shape[0]
    cb = imgback.shape[1]
    hrb=round(rb/2)
    hcb=round(cb/2)
    hr=round(r/2)
    hc=round(c/2)

    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)
    imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]

    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))

    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)

    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)

    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)

    return imgpaste

def cosineTheorem(Lidar, radian1, radian2):
    theta = abs(radian1-radian2)
    distance = Lidar[radian1][1] ** 2 + Lidar[radian2][1] ** 2 - 2 * Lidar[radian1][1] * Lidar[radian2][1] * math.cos(abs(radian2 - radian1))

    return distance

def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
# def beep(freq, dur=100):
#         winsound.Beep(freq, dur)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

if __name__ == '__main__':
    #学習前YOLO
    # cfgfile = "cfg/yolov3.cfg" # 設定ファイル
    # weightsfile = "weight/yolov3.weights" # 重みファイル
    # classes = load_classes('data/coco.names') # 識別クラスのリスト

    #マスク学習後YOLO
    cfgfile = "cfg/mask.cfg" # 設定ファイル
    weightsfile = "weight/mask_1500.weights" # 重みファイル
    classes = load_classes('data/mask.names') # 識別クラスのリスト
    args = arg_parse() # 引数を取得
    confidence = float(args.confidence) # 信頼性の設定値を取得
    nms_thesh = float(args.nms_thresh) # 閾値を取得
    start = 0
    CUDA = torch.cuda.is_available() # CUDAが使用可能かどうか

    num_classes = 2 # クラスの数
    bbox_attrs = 5 + num_classes
    max = 1 #限界人数
    WIDTH = 480
    HEIGHT = 360
    FPS = 30
    RADIAN = 100
    dif_radian = 2
    portion = 1.0
    num_camera = 4 #camera数
    model = [[] for i in range(num_camera)]
    inp_dim = [[] for i in range(num_camera)]
    cap = [[] for i in range(num_camera)]
    ret = [[] for i in range(num_camera)]
    frame = [[] for i in range(num_camera)]
    img = [[] for i in range(num_camera)]
    orig_im = [[] for i in range(num_camera)]
    dim = [[] for i in range(num_camera)]
    im_dim = [[] for i in range(num_camera)]
    # output = [[] for i in range(num_camera)]
    # output = torch.tensor(output)
    # print("output_shape\n", output.shape)

    for i in range(num_camera):
        model[i] = Darknet(cfgfile) #model1の作成
        model[i].load_weights(weightsfile) # model1に重みを読み込む

        model[i].net_info["height"] = args.reso
        inp_dim[i] = int(model[i].net_info["height"])

        assert inp_dim[i] % 32 == 0
        assert inp_dim[i] > 32

    #mixer.init()        #初期化

    if CUDA:
        for i in range(num_camera):
            model[i].cuda() #CUDAが使用可能であればcudaを起動
        print("GPU!!!!!")
    for i in range(num_camera):
        model[i].eval()
	
    cap[0] = cv2.VideoCapture(0) #カメラを指定（USB接続）
    cap[0].set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap[0].set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap[0].set(cv2.CAP_PROP_FPS, FPS)

    cap[1] = cv2.VideoCapture(1) #カメラを指定（USB接続）
    cap[1].set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap[1].set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap[1].set(cv2.CAP_PROP_FPS, FPS)

    cap[2] = cv2.VideoCapture(2) #カメラを指定（USB接続）
    cap[2].set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap[2].set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap[2].set(cv2.CAP_PROP_FPS, FPS)

    cap[3] = cv2.VideoCapture(3) #カメラを指定（USB接続）
    cap[3].set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap[3].set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap[3].set(cv2.CAP_PROP_FPS, FPS)

    img1 = cv2.imread("images/phase_1.jpg")
    img2 = cv2.imread("images/phase_2.jpg")
    img3 = cv2.imread("images/phase_2_red.jpg")
    img4 = cv2.imread("images/phase_3.jpg")
    
    frames = 0
    count_frame = 0 #フレーム数カウント
    flag = 0 #密状態（0：疎密，1：密入り）
    start = time.time()
    print('start')
    while (cap[i].isOpened() for i in range(num_camera)): #カメラが起動している間
        with open('/home/jphacks2020/LiDAR.csv', 'r', encoding="utf-8_sig", newline = '') as f:
            l = csv.reader(f)
            LiDAR = [row for row in l]
        if len(LiDAR) == 0:
            continue
        LiDAR_array = np.array(LiDAR)
        LiDAR_array = np.delete(LiDAR_array, [-1],0)
        # if LiDAR_array[-1] == ['', '']:
        #     LiDAR_array = np.delete(LiDAR_array, [-1],0)
        for i in range(len(LiDAR_array)):
            if len(LiDAR_array)>i:
                if len(LiDAR_array[i]) != 3:
                    LiDAR_array = np.delete(LiDAR_array, [i],0)
                elif LiDAR_array[i][0] == '' or LiDAR_array[i][1] == '':
                    LiDAR_array = np.delete(LiDAR_array, [i],0)
        print(LiDAR_array)
        while len(LiDAR_array) < 360:
            empty_array = [["360", "inf", "\n"]]
            LiDAR_array = np.append(LiDAR_array, empty_array, 0)

       
        # print(LiDAR_array)
        print(len(LiDAR_array))
        LiDAR_array1, LiDAR_array2, bug= np.split(LiDAR_array, 3, 1)

        for i in range(len(LiDAR_array1)):
            if LiDAR_array1[i] == "inf":
                LiDAR_array1[i] = "-1"
        LiDAR_array1_int = LiDAR_array1.astype(np.float32)
        LiDAR_array1_int = LiDAR_array1_int.astype(np.int64)
        LiDAR_array2_float = LiDAR_array2.astype(np.float32)
        # print("LiDAR_array_1int",LiDAR_array1_int)
        # print("LiDAR_array_2float",LiDAR_array2_float)
        LiDAR_array_cast = np.append(LiDAR_array1_int, LiDAR_array2_float, axis = 1)
        LiDAR_list = LiDAR_array_cast.tolist()

        print("LiDAR_len", type(LiDAR[0][0]))
        count=0 #人数をカウント
        point = [[] for i in range(num_camera)]
        
        for i in range(num_camera):
            ret[i], frame[i] = cap[i].read()
            #print(cap[i])
            #print(frame[i])

        #print("len_frame", len(frame))
        #print(frame)
        #if (1):
        # 解析準備としてキャプチャ画像を加工
        for i in range(num_camera):
            img[i], orig_im[i], dim[i] = prep_image(frame[i], inp_dim[i])
            im_dim[i] = torch.FloatTensor(dim[i]).repeat(1,2)

        if CUDA:
            for i in range(num_camera):
                im_dim[i] = im_dim[i].cuda()
                img[i] = img[i].cuda()
       
        output0 = model[0](Variable(img[0]), CUDA)
        output1 = model[1](Variable(img[1]), CUDA)
        output2 = model[2](Variable(img[2]), CUDA)
        output3 = model[3](Variable(img[3]), CUDA)

        output0 = write_results(output0, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        output1 = write_results(output1, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        output2 = write_results(output2, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        output3 = write_results(output3, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        #print("output", 0, ":\n", output0)
        #print(output0.shape)
        
        output0[:,1:5] = torch.clamp(output0[:,1:5], 0.0, float(inp_dim[0]))/inp_dim[0]
        im_dim[0] = im_dim[0].repeat(output0.size(0), 1)
        output0[:,[1,3]] *= frame[0].shape[1]
        output0[:,[2,4]] *= frame[0].shape[0]

        output1[:,1:5] = torch.clamp(output1[:,1:5], 0.0, float(inp_dim[1]))/inp_dim[1]
        im_dim[1] = im_dim[1].repeat(output1.size(0), 1)
        output1[:,[1,3]] *= frame[1].shape[1]
        output1[:,[2,4]] *= frame[1].shape[0]

        output2[:,1:5] = torch.clamp(output2[:,1:5], 0.0, float(inp_dim[2]))/inp_dim[2]
        im_dim[2] = im_dim[2].repeat(output2.size(0), 1)
        output2[:,[1,3]] *= frame[2].shape[1]
        output2[:,[2,4]] *= frame[2].shape[0]

        output3[:,1:5] = torch.clamp(output3[:,1:5], 0.0, float(inp_dim[3]))/inp_dim[3]
        im_dim[3] = im_dim[3].repeat(output3.size(0), 1)
        output3[:,[1,3]] *= frame[3].shape[1]
        output3[:,[2,4]] *= frame[3].shape[0]

        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x0: write(x0, orig_im[0], 0), output0))
        list(map(lambda x1: write(x1, orig_im[1], 1), output1))
        list(map(lambda x2: write(x2, orig_im[2], 2), output2))
        list(map(lambda x3: write(x3, orig_im[3], 3), output3))
        print("count:\n",count)
        print("count_frame", count_frame)
        # print("framex", frame[0].shape[1])
        # print("framey", frame[0].shape[0])
        #print("point0",point[0])
        #print("type_point", point[0][0][0])
        #print("len_LiDAR", len(LiDAR))

        #LiDARの情報の人識別
        radian_lists = []
        radian_list = []
        close_list = [0] * 4
        dense_list = [0] * 4
        num_person = [0] * 4
        for k, (radian, length) in enumerate(LiDAR_list):
            #print("k", k)
            if not length == -1:
                #print("radian:", radian)
                #print("length:", length)
                if k % 90 == 0:
                    if not k == 0:
                        radian_lists.append(radian_list)
                        radian_list = []
                if k < 90:
                    radian_cam = []
                    for num in range(len(point[0])):
                        # print("num:",num)
                        # print("p",point[0])
                        # print("p_len", len(point[0]))
                        if len(point[0]) != 0:
                        # print("num", num)
                        # print("p1", p[0])
                        # print("frame_shape", frame[0].shape)
                            if point[0][num][1] < frame[0].shape[1] * portion:
                                radian_cam.append(point[0][num][0] / frame[0].shape[1] * RADIAN)
                                for dif in range(dif_radian*2):
                                    for radi_num in range(len(radian_cam)):
                                        # print("radian_cam", radian_cam[radi_num])
                                        if radian_cam[radi_num] == '':
                                            radian_cam[radi_num] = 400
                                        # print("radian_cam:",radian_cam)
                                        if int(radian)+dif-dif_radian == int(radian_cam[radi_num]):
                                            num_person[0] += 1
                                            radian_list.append(radian)
                       
                elif k < 180:
                    radian_cam = [] 
                    for num in range(len(point[1])):
                        # print("num:",num)
                        # print("p",point[1])
                        # print("p_len", len(point[1]))
                        if len(point[1][num]) != 0:
                            if point[1][num][1] < frame[1].shape[1] * portion:
                                radian_cam.append(point[1][num][0] / frame[1].shape[1] * RADIAN)
                                for dif in range(dif_radian*2):
                                    for radi_num in range(len(radian_cam)):
                                        # print("radian_cam", radian_cam[radian_cam])
                                        if radian_cam[radi_num] == '':
                                            radian_cam[radi_num] = 400
                                        # print("radian_cam:",radian_cam)
                                        if int(radian)+dif-dif_radian == int(radian_cam[radi_num]):
                                            num_person[1] += 1
                                            radian_list.append(radian)
                        
                elif k < 270:
                    #radian_cam = [[] for i in range(len(point[2]))]
                    radian_cam = []
                    for num in range(len(point[2])):
                        # print("num:",num)
                        # print("p",point[2])
                        # print("p_len", len(point[2]))
                        if len(point[2][num]) != 0:
                            if point[2][num][1] < frame[1].shape[1] * portion:
                                #radian_cam[num] = point[2][num][0] / frame[2].shape[1] * RADIAN
                                radian_cam.append(point[2][num][0] / frame[2].shape[1] * RADIAN)
                                for dif in range(dif_radian):
                                    for radi_num in range(len(radian_cam)):
                                        # print("radian_cam", radian_cam[radian_cam])
                                        if radian_cam[radi_num] == '':
                                            radian_cam[radi_num] = 400
                                        # print("radian_cam:",radian_cam)
                                        if int(radian+2)+dif-dif_radian == int(radian_cam[radi_num]):
                                            num_person[2] += 1
                                            radian_list.append(radian)
                        
                else:
                    radian_cam = []
                    for num in range(len(point[3])):
                        # print("num:",num)
                        # print("p",point[3])
                        # print("p_len", len(point[3]))
                        if len(point[3][num]) != 0:
                            if point[3][num][1] < frame[1].shape[1] * portion:
                                radian_cam.append(point[3][num][0] / frame[3].shape[1] * RADIAN)
                                for dif in range(dif_radian*2):
                                    for radi_num in range(len(radian_cam)):
                                        # print("radian_cam", radian_cam[radian_cam])
                                        if radian_cam[radi_num] == '':
                                            radian_cam[radi_num] = 400
                                        # print("radian_cam:",radian_cam)
                                        if int(radian)+dif-dif_radian == int(radian_cam[radi_num]):
                                            num_person[3] += 1
                                            radian_list.append(radian)
                       
        radian_lists.append(radian_list)
        # print("radian_lists_len", len(radian_lists))
        print("len_radian_lists", len(radian_lists))

        #距離計算
        dis_list = []
        distance = []
        for direction in range(4):
            if len(radian_lists[direction]) > 1:
                # n = combinations_k(len(radian_lists[direction]), 2)
                dis_combination = list(itertools.combinations(radian_lists[direction], 2))
                distance = [[] for i in range(len(dis_combination))]
                #print(type(LiDAR_list[0][0]))
                for num_dis, com_list in enumerate(dis_combination):
                    distance[num_dis] = cosineTheorem(LiDAR_list, int(com_list[0]), int(com_list[1]))
            dis_list.append(distance)

        dense_flag = [0] * 4
        #密集判定
        for direction in range(4):
            close = 0 #密接数
            dense = 0 #密集数
            #print(type(direction))
            for dis in dis_list[direction]:
                if dis < 1.0:
                    close += 1
                    close_list[direction] = 1
                    dense_flag[direction] = 1
            if num_person[direction] > 2:
                dense += 1
                dense_list[direction] = 1
                dense_flag[direction] = 2
            print("direction_people",direction,":",num_person[direction])
            print("direction_close", direction, ":", close)
            print("direction_dense", direction, ":", dense)
        print("close_list", close_list)
        print("dense_list", dense_list)

        # print("point1",point[1])


    
        if 1 in dense_flag:
            x=0
            y=0
            angle=20
            scale=1.5
            count_frame = 0
            for i in range(num_camera):
                print("direction,state:",i,",",dense_flag[i])

                if close_list[i] == 1:
                    orig_im[i] = cvpaste(img1, orig_im[i], x, y, angle, scale)
        if 2 in dense_flag:
            x=-30
            y=10
            angle=20
            scale=1.1
            count_frame += 1
            if dense_flag[i] == 2:
                if count_frame%2==1:
                    for i in range(num_camera):
                        orig_im[i] = cvpaste(img2, orig_im[i], x, y, angle, scale)
                else:
                    for i in range(num_camera):
                        orig_im[i] = cvpaste(img3, orig_im[i], x, y, angle, scale)
        else:
            count_frame = 0


        #googlehome
        if 1 in close_list:
            print("close")
            # ifttt_demo2.googleDense()
            t2=time.time()
            if count3==0:
                flag2=1
            elif t2-t1>=5:
                flag2=1

            if flag2==1:
                ifttt_demo_log.google_sheet()
                ifttt_demo_log.ifttt_webhook(3)
                play.googlehome()
                # ifttt_demo2.googleDense()
                t1=time.time()
            flag2=0
            count3+=1
        if 1 in dense_list:
            print("dense")
            #ifttt_demo_log.ifttt_webhook(2)
            t2=time.time()
            if count3==0:
                flag2=1
            elif t2-t1>=5:
                flag2=1

            if flag2==1:
                ifttt_demo_log.google_sheet()
                ifttt_demo_log.ifttt_webhook(2)
                play.googlehome()
                # ifttt_demo2.googleDense()
                t1=time.time()
            flag2=0
            count3+=1



        im_h_resize1 = hconcat_resize_min([orig_im[0],orig_im[1]])
        im_h_resize2 = hconcat_resize_min([orig_im[2],orig_im[3]])
        im_v_resize = vconcat_resize_min([im_h_resize1,im_h_resize2])
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", im_v_resize )
        key = cv2.waitKey(1)
        # qキーを押すと動画表示の終了
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("count_frame:\n", count_frame)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
