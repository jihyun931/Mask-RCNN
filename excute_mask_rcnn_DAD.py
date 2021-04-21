# execute mask_rcnn
# save txt file
# output : " frame_id, -1, x1, y1, w, h, scores, -1, -1, -1 "<- deep sort want this format

import torch
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print("Error: Creating directory." + directory)

def vis_bbox(boxes_arr, image, save_path):
    draw = ImageDraw.Draw(image)
    num_bboxes = boxes_arr.shape[0]

    for i in range(num_bboxes):
        [x1,y1,x2,y2] = boxes_arr[i]
        draw.rectangle(((x1,y1),(x2,y2)), outline=(0,255,0), width=2)

    image.save(save_path)




#choose the pretrained model.
config_file = "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
#Load the model
coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7,)

dataset_path = "E:/DAD/frames/"

main_folder_list = os.listdir(dataset_path) #[test, train]

for main_folder in main_folder_list[1:]:
    sub_folder_list = os.listdir(dataset_path + main_folder) #[negative, positive]
    
    for sub_folder in sub_folder_list[1:]:
        print("====================================")
        print( main_folder, sub_folder)

        folder_list = os.listdir(dataset_path + main_folder +"/" + sub_folder)

        createFolder("E:/DAD/mask_rcnn_result/" + main_folder +"/" + sub_folder)

        for foldername in folder_list[:10]:
            path = dataset_path + main_folder +"/" + sub_folder + "/" + foldername
            print(">> Target data folder: ", path)
            file_list = os.listdir(path)
            file_list = [file for file in file_list if file.endswith(".jpg")]
            print(" Number of frames:", len(file_list))
        
            trigger = 0
        
            for i in range(len(file_list)):
                image_path = path+'/'+file_list[i]
                image_original = Image.open(image_path)
                image = np.array(image_original)[:,:,[2,1,0]]
                
                bbox_predictions = coco_demo.compute_prediction(image)
                top_predictions = coco_demo.select_top_predictions(bbox_predictions)
                
                boxes = top_predictions.bbox
                labels = top_predictions.get_field("labels").tolist()
                scores = top_predictions.get_field("scores").tolist()
                
                boxes_arr = np.array(boxes) #[x1,y1,x2,y2]

                #--------- Visualization -------------------
                #save_img_path = "D:/DAD/detection_check/train_positive/" + foldername +"/" + file_list[i]
                #createFolder("D:/DAD/detection_check/train_positive/" + foldername)
                #vis_bbox(boxes_arr, image_original, save_img_path)
                #------------------------------------------

                # start frame is 1
                frame_id = i+1
        
                for k in range(len(scores)):
                    label_id = labels[k]
                    # Only person(1), bicycle(2), car(3), motorcycle(4), bus(6), truck(8)
                    if label_id == 1 or label_id==2 or label_id==3 or label_id==4 or label_id==6 or label_id==8 :
                        if trigger == 0 : #if i==0 and k==0:
                            data = np.array([frame_id, -1, boxes_arr[k][0], boxes_arr[k][1], boxes_arr[k][2]-boxes[k][0], boxes_arr[k][3]-boxes_arr[k][1],scores[k], -1, -1, -1])
                            trigger = 1
                        else :
                            data_temp = np.array([frame_id, -1, boxes_arr[k][0], boxes_arr[k][1], boxes_arr[k][2]-boxes[k][0], boxes_arr[k][3]-boxes_arr[k][1],scores[k], -1, -1, -1])
                            data = np.vstack([data,data_temp])
            
            save_path = "E:/DAD/mask_rcnn_result/" + main_folder +"/" + sub_folder +"/" + foldername +".txt"
            np.savetxt(save_path, data, fmt='%.3f', delimiter=',')
            print(" Save path: ", save_path)