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

#from pygame import mixer
#import winsound


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
    print("label:", label)
    # 人数カウント
    if(label=='person'):
        count+=1
    print(count)

    return count

def write(x, img):
    global count
    # 画像に結果を描画
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    print("label:\n", label)
    # 人数カウント
    if(label=='person'):
        count+=1
    print(count)

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

# def beep(freq, dur=100):
#         winsound.Beep(freq, dur)

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg" # 設定ファイル
    weightsfile = "weight/yolov3.weights" # 重みファイル
    num_classes = 80 # クラスの数

    args = arg_parse() # 引数を取得
    confidence = float(args.confidence) # 信頼性の設定値を取得
    nms_thesh = float(args.nms_thresh) # 閾値を取得
    start = 0
    CUDA = torch.cuda.is_available() # CUDAが使用可能かどうか

    num_classes = 80 # クラスの数
    bbox_attrs = 5 + num_classes
    max = 3 #限界人数

    model1 = Darknet(cfgfile) #model1の作成
    model1.load_weights(weightsfile) # model1に重みを読み込む
    model2 = Darknet(cfgfile) #model1の作成
    model2.load_weights(weightsfile) # model1に重みを読み込む


    model2.net_info["height"] = args.reso
    inp_dim = int(model2.net_info["height"])
    im_dim = [[] for i in range(num_camera)]

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #mixer.init()        #初期化

    if CUDA:
        model1.cuda() #CUDAが使用可能であればcudaを起動
        model2.cuda() #CUDAが使用可能であればcudaを起動

    num = 1 #camera数
    model1.eval()
    model2.eval()
    cap1 = cv2.VideoCapture(1) #カメラを指定（USB接続）
    #cap2 = cv2.VideoCapture(0) #カメラを指定（USB接続）
    # cap = cv2.VideoCapture("movies/sample.mp4")
    #cap = cv2.VideoCapture("movies/one_v2.avi")

    # Use the next line if your camera has a username and password
    # cap = cv2.VideoCapture('protocol://username:password@IP:port/1')
    #cap = cv2.VideoCapture('rtsp://admin:admin@192.168.11.4/1') #（ネットワーク接続）
    #cap = cv2.VideoCapture('rtsp://admin:admin@192.168.11.4/80')
    #cap = cv2.VideoCapture('http://admin:admin@192.168.11.4:80/video')
    #cap = cv2.VideoCapture('http://admin:admin@192.168.11.4/camera-cgi/admin/recorder.cgi?action=start&id=samba')
    #cap = cv2.VideoCapture('http://admin:admin@192.168.11.4/recorder.cgi?action=start&id=samba')
    #cap = cv2.VideoCapture('http://admin:admin@192.168.11.5:80/snapshot.jpg?user=admin&pwd=admin&strm=0')
    print('-1')

    #assert cap.isOpened(), 'Cannot capture source' #カメラが起動できたか確認

    img1 = cv2.imread("images/phase_1.jpg")
    img2 = cv2.imread("images/phase_2.jpg")
    img3 = cv2.imread("images/phase_2_red.jpg")
    img4 = cv2.imread("images/phase_3.jpg")
    #mixer.music.load("voice/voice_3.m4a")
    #print(img1)
    frames = 0
    count_frame = 0 #フレーム数カウント
    start = time.time()
    print('-1')
    while cap1.isOpened(): #カメラが起動している間
        count=0 #人数をカウント

        ret, frame = cap1.read() #キャプチャ画像を取得
        if ret:
            # 解析準備としてキャプチャ画像を加工
            img, orig_im, dim = prep_image(frame, inp_dim)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model1(Variable(img), CUDA)
            print("output:\n", output)
            print(output.shape)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            # print("output:\n", output)

            # FPSの表示
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

                # qキーを押すとFPS表示の終了
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            print("len_output", len(output))
            print("len_output[]", len(output[0]))
            # print("len_output[][]", len(output[0][0]))
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            classes = load_classes('data/coco.names') # 識別クラスのリスト
            colors = pkl.load(open("pallete", "rb"))

            #count = lambda x: count(x, orig_im, count) #人数をカウント

            list(map(lambda x: write(x, orig_im), output))
            print("count:\n",count)

            if count > max:
                count_frame += 1
                #print("-1")
                if count_frame <= 50:
                    x=0
                    y=0
                    angle=20
                    scale=1.5
                    imgpaste = cvpaste(img1, orig_im, x, y, angle, scale)
                    #mixer.music.play(1)
                    # 2000Hzで500ms秒鳴らす
                    #beep(2000, 500)
                elif count_frame <= 100:
                    x=-30
                    y=10
                    angle=20
                    scale=1.1
                    if count_frame%2==1:
                        imgpaste = cvpaste(img2, orig_im, x, y, angle, scale)
                    else:
                        imgpaste = cvpaste(img3, orig_im, x, y, angle, scale)
                else:
                    x=-30
                    y=0
                    angle=20
                    scale=1.5
                    imgpaste = cvpaste(img4, orig_im, x, y, angle, scale)
                    if count_frame > 101: #<--2フレームずらす
                        print("\007") #警告音
                        time.sleep(3)
                cv2.imshow("frame", imgpaste)
            else:
                count_frame = 0
                #print("-2")
                cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            # qキーを押すと動画表示の終了
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("count_frame:\n", count_frame)
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        else:
            break

