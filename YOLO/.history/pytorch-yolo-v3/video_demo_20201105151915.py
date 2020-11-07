from __future__ import division
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
# import play

#from pygame import mixer
#import winsound

camera_name = {
    "north":0,
    "south":2,
    "east":1,
    "west":3,
}


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
    label = "{0}".format(classes[cls])
    print("label:", label)
    # 人数カウント
    if(label=='no-mask'):
        count+=1
    print(count)

    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

    p[0] = (c2[0]-c1[0])/2
    p[1] = (c2[1]-c1[1])/2
    point[camId].append(p)
    print("point0",point[0])
    print("point1",point[1])

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

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

# def beep(freq, dur=100):
#         winsound.Beep(freq, dur)

if __name__ == '__main__':
    #学習前YOLO
    # cfgfile = "cfg/yolov3.cfg" # 設定ファイル
    # weightsfile = "weight/yolov3.weights" # 重みファイル
    # classes = load_classes('data/coco.names') # 識別クラスのリスト

    #マスク学習後YOLO
    cfgfile = "cfg/mask.cfg" # 設定ファイル
    weightsfile = "weight/mask_1500.weights" # 重みファイル
    classes = load_classes('data/mask.names') # 識別クラスのリスト


    num_classes = 80 # クラスの数

    args = arg_parse() # 引数を取得
    confidence = float(args.confidence) # 信頼性の設定値を取得
    nms_thesh = float(args.nms_thresh) # 閾値を取得
    start = 0
    CUDA = torch.cuda.is_available() # CUDAが使用可能かどうか

    num_classes = 80 # クラスの数
    bbox_attrs = 5 + num_classes
    max = 0 #限界人数
    num_camera = 2 #camera数
    model = [[] for i in range(num_camera)]
    inp_dim = [[] for i in range(num_camera)]
    cap = [[] for i in range(num_camera)]
    ret = [[] for i in range(num_camera)]
    frame = [[] for i in range(num_camera)]
    img = [[] for i in range(num_camera)]
    orig_im = [[] for i in range(num_camera)]
    dim = [[] for i in range(num_camera)]
    output0 = []
    output1 = []
    output2 = []
    output3 = []

    point = [[] for i in range(num_camera)]
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

    for i in range(num_camera):
        model[i].eval()

    cap[0] = cv2.VideoCapture(0) #カメラを指定（USB接続）
    cap[1] = cv2.VideoCapture(1) #カメラを指定（USB接続）
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
    flag = 0 #密状態（0：疎密，1：密入り）
    start = time.time()
    print('-1')
    while (cap[i].isOpened() for i in range(num_camera)): #カメラが起動している間
        count=0 #人数をカウント
        point = [[] for i in range(num_camera)]
        for i in range(num_camera):
            ret[i], frame[i] = cap[i].read() #キャプチャ画像を取得
        if (ret[i] for i in range(num_camera)):
            # 解析準備としてキャプチャ画像を加工
            for i in range(num_camera):
                img[i], orig_im[i], dim[i] = prep_image(frame[i], inp_dim[i])

            if CUDA:
                for i in range(num_camera):
                    im_dim[i] = im_dim[i].cuda()
                    img[i] = img[i].cuda()

            # for i in range(num_camera):
                # output[i] = model[i](Variable(img[i]), CUDA)
            output0 = model[0](Variable(img[0]), CUDA)
            output1 = model[1](Variable(img[1]), CUDA)
            # output2 = model[2](Variable(img[2]), CUDA)
            # output3 = model[3](Variable(img[3]), CUDA)

                #print("output:\n", output)
                # output[i] = write_results(output[i], confidence, num_classes, nms = True, nms_conf = nms_thesh)
            output0 = write_results(output0, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            output1 = write_results(output1, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            # output2 = write_results(output2, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            # output3 = write_results(output3, confidence, num_classes, nms = True, nms_conf = nms_thesh)
                # print("output", i, ":\n", output[i])
                # print(output.shape)
            """
            # FPSの表示
            if (type(output[i]) == int for i in range(num_camera)):
                print("表示")
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

                # qキーを押すとFPS表示の終了
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            for i in range(num_camera):
                output[i][:,1:5] = torch.clamp(output[i][:,1:5], 0.0, float(inp_dim[i]))/inp_dim[i]
                output[i][:,[1,3]] *= frame[i].shape[1]
                output[i][:,[2,4]] *= frame[i].shape[0]
            """
            # # FPSの表示
            # if type(output0) == int:
            #     print("表示")
            #     frames += 1
            #     print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            #     # qキーを押すとFPS表示の終了
            #     key = cv2.waitKey(1)
            #     if key & 0xFF == ord('q'):
            #         break
            #     continue
            # for i in range(num_camera):
            output0[:,1:5] = torch.clamp(output0[:,1:5], 0.0, float(inp_dim[i]))/inp_dim[i]
            output0[:,[1,3]] *= frame[0].shape[1]
            output0[:,[2,4]] *= frame[0].shape[0]

            output1[:,1:5] = torch.clamp(output1[:,1:5], 0.0, float(inp_dim[i]))/inp_dim[i]
            output1[:,[1,3]] *= frame[1].shape[1]
            output1[:,[2,4]] *= frame[1].shape[0]

            # output2[:,1:5] = torch.clamp(output2[:,1:5], 0.0, float(inp_dim[i]))/inp_dim[i]
            # output2[:,[1,3]] *= frame[i].shape[1]
            # output2[:,[2,4]] *= frame[i].shape[0]

            # output3[:,1:5] = torch.clamp(output3[:,1:5], 0.0, float(inp_dim[i]))/inp_dim[i]
            # output3[:,[1,3]] *= frame[i].shape[1]
            # output3[:,[2,4]] *= frame[i].shape[0]



            colors = pkl.load(open("pallete", "rb"))

            #count = lambda x: count(x, orig_im, count) #人数をカウント
            """
            for i in range(num_camera):
                list(map(lambda x: write(x, orig_im[i]), output[i]))
            print("count:\n",count)
            """
            # for i in range(num_camera):
            #     list(map(lambda x: write(x, orig_im[i]), output))
            list(map(lambda x0: write(x0, orig_im[0],0), output0))
            list(map(lambda x1: write(x1, orig_im[1],1), output1))
            # print("x0",x0)
            # list(map(lambda x2: write(x2, orig_im[2],2), output2))
            # list(map(lambda x3: write(x3, orig_im[3],3), output3))
            # print("point0",point[0])
            # print("point1",point[1])


            print("count:\n",count)
            print("count_frame", count_frame)


            if count > max:
                count_frame += 1
                #print("-1")
                if count_frame <= 50:
                    x=0
                    y=0
                    angle=20
                    scale=1.5
                    for i in range(num_camera):
                        imgpaste = cvpaste(img1, orig_im[i], x, y, angle, scale)
                    if flag == 1:
                        # play.googlehome()
                        flag += 1
                    #mixer.music.play(1)
                elif count_frame <= 100:
                    x=-30
                    y=10
                    angle=20
                    scale=1.1
                    if count_frame%2==1:
                        for i in range(num_camera):
                            imgpaste = cvpaste(img2, orig_im[i], x, y, angle, scale)
                    else:
                        for i in range(num_camera):
                            imgpaste = cvpaste(img3, orig_im[i], x, y, angle, scale)
                    if flag == 2:
                        # play.googlehome()
                        flag += 1
                else:
                    x=-30
                    y=0
                    angle=20
                    scale=1.5
                    for i in range(num_camera):
                        imgpaste = cvpaste(img4, orig_im[i], x, y, angle, scale)
                    if count_frame > 101: #<--2フレームずらす
                        print("\007") #警告音
                        time.sleep(3)
                    if flag == 3:
                        # play.googlehome()
                        flag += 1
                # cv2.imshow("frame", imgpaste)
            else:
                count_frame = 0
                flag = 0
                #print("-2")
                # for i in range(num_camera):
            im_h_resize = hconcat_resize_min(orig_im)
            cv2.imshow("frame", im_h_resize )
                # play.googlehome()
            key = cv2.waitKey(1)
            # qキーを押すと動画表示の終了
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("count_frame:\n", count_frame)
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        else:
            break

