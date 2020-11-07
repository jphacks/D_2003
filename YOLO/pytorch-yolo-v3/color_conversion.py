import numpy as np
import imutils

import cv2

img_rgb = cv2.imread('images/phase_2.jpg')

Conv_hsv_Gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)

#img_rgb.Set(mask,cv2.Scalar(0,0,255))
img_rgb[mask == 255] = [0, 0, 255]

cv2.imwrite("images/imgOriginal.jpg", img_rgb)  # show windows


cv2.waitKey(0)