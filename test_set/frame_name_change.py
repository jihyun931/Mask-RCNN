import cv2
import numpy as np
import glob
import os
 
dir_path = "08/images/origin"
image_list = os.listdir(dir_path)
image_list = [file for file in image_list if file.endswith(".jpg")]

for i in range(len(image_list)):
    image = cv2.imread(dir_path+"/"+image_list[i])
    k = i+1
    nth = "{:03d}".format(k)
    save_path = "08/images/"+str(nth)+".jpg"
    cv2.imwrite(save_path, image)