import os
import random
from numpy import append
import torch
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
from tqdm import tqdm
from PIL import Image as im
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Parameters for model extraction
yolo_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.7,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

# Classes
classes = load_classes(yolo_params["class_path"])

# Load the model with weights
detectron = YOLOv3Predictor(params=yolo_params)

# Path to similar recommender file and path to complete the look recommender file
from_path = '../Output/Cropped'
input_path = '../recommnder_input'
output_path = '../Output-recommend'
output_path_similar = '{}/similar'.format(output_path)
output_path_complete = '{}/complete'.format(output_path)

# Check if there is the required path, if not exist - will make a directory
if not os.path.isdir(output_path):
        os.mkdir(output_path)
if not os.path.isdir(output_path_similar):
        os.mkdir(output_path_similar)
if not os.path.isdir(output_path_complete):
        os.mkdir(output_path_complete)

# Polygon color picker
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])


def predict_single_image(image_name, path):
    class_predict = []
    new_path = '{}/{}'.format(input_path, image_name)
    img = cv2.imread(new_path)
    detections = detectron.get_detections(img)
    if len(detections) != 0:
        # Sort detections
        detections.sort(reverse=False ,key = lambda x:x[4])
        # Loop for each class detected in the image
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            if classes[int(cls_pred)] in ["scarf", "belt"]:
                continue
            class_predict.append(classes[int(cls_pred)])        
            color = colors[int(cls_pred)]
            
            color = tuple(c*255 for c in color)
            color = (.7*color[2],.7*color[1],.7*color[0])       
                
            font = cv2.FONT_HERSHEY_SIMPLEX   
            # Polygon pixels
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Text for polygon
            text =  "{} ({}%)".format(classes[int(cls_pred)] ,int(cls_conf * 100))
            
            # Fix pixels of polygon for text input
            cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
            y1 = 0 if y1<0 else y1
            y1_rect = y1-25
            y1_text = y1-5
            if y1_rect<0:
                y1_rect = y1+27
                y1_text = y1+20
            
            # Create the polygon on img
            cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
            cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        img_id = new_path.split('/')[-1].split('.')[0]
        cv2.imwrite('{}/{}/{}.jpg'.format(path, img_id, img_id),img)
    return class_predict
            
# Similar items           
for f in os.listdir('{}'.format(input_path)):
    path = f.removesuffix('.jpg')
    if not os.path.isdir("{}/{}".format(output_path_similar, path)):
        os.mkdir("{}/{}".format(output_path_similar, path))
    for item_class in predict_single_image('{}/{}'.format(input_path, f), output_path_similar):
        items = os.listdir('{}/{}'.format(from_path, item_class))
        rand = random.randint(1, len(items) - 1)
        shutil.copyfile('{}/{}/{}'.format(from_path, item_class, items[rand]), '{}/{}/similar_{}.jpg'.format(output_path_similar, path, item_class))


# Complete the look        
for f in os.listdir('{}'.format(input_path)):
    path = f.removesuffix('.jpg')
    if not os.path.isdir("{}/{}".format(output_path_complete, path)):
        os.mkdir("{}/{}".format(output_path_complete, path))
    for item_class in predict_single_image('{}/{}'.format(input_path, f), output_path_complete):
        flag = True
        checker = { 1: ['footwear', 'boots'], 2: ['pants', 'shorts', 'dress', 'skirt'], 3: ['top' , 'dress', 'outer'] }
        rand_class = ""
        while(rand_class == item_class or flag):
            flag = False
            rand_class = random.choice(classes)
            if rand_class in checker[1] and item_class in checker[1]:
               flag = True
            elif rand_class in checker[2] and item_class in checker[2]:
               flag = True
            elif rand_class in checker[3] and item_class in checker[3]:
               flag = True
        items = os.listdir('{}/{}'.format(from_path, rand_class))
        rand = random.randint(1, len(items) - 1)
        shutil.copyfile('{}/{}/{}'.format(from_path, rand_class, items[rand]), "{}/{}/complete_{}_for_{}.jpg".format(output_path_complete, path, rand_class, item_class))