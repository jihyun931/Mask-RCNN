
import cv2
import numpy as np
import glob

import os

#img_array = []
#for filename in glob.glob('test_set/08/images/*.jpg'):
#    img = cv2.imread(filename)
#    height, width, layers = img.shape
#    size = (width,height)
#    img_array.append(img)
#
#out = cv2.VideoWriter('test_set/08/test_video_08.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
#
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()


path = "C:/Users/user/Desktop/Anomaly_Detection/IROS2019/tad-IROS2019-master/A3D/dataset/"
for foldername in os.listdir(path):
    print(foldername)
    img_array = []
    for filename in glob.glob(path+foldername+"/images/*.jpg"):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    
    out = cv2.VideoWriter(path+foldername+"/test_video.avi",cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()