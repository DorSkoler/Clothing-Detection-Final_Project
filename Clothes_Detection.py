# imports
from numpy import append
import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
from tqdm import tqdm
from PIL import Image as im

# Input / output paths
output_path = '../'
input_path = '../Data'

# Device check, can run with gpu for faster performance
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

# Polygon color picker
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])

# Load the model with weights
detectron = YOLOv3Predictor(params=yolo_params)

# Making directories for model output
all_dir = ["", "/Cropped", "/Full"]

for dir in all_dir:
    if not os.path.isdir('{}/Output{}'.format(output_path, dir)):
        os.mkdir('{}/Output{}'.format(output_path, dir)) 

for cls in  classes:
    if os.path.isdir('{}/Output/Cropped/{}'.format(output_path, cls)):
        continue
    os.mkdir('{}/Output/Cropped/{}'.format(output_path, cls))
 
# Main function
dir_len = len(os.listdir(input_path))
image_num = 0
read_paths = []
# Txt file for all output accuracy
# Accuracy log file
f = open('{}/Output/accuracy.txt'.format(output_path), 'a')

# Main loop for all images in input folder
for img_path in os.listdir(input_path):
    # Counter for images
    image_num+=1
    # Build path to specific image
    new_path = '{}/{}'.format(input_path, img_path)
    
    # Check if image was already detected
    if new_path in read_paths:
        continue
    
    # Adding the image path to the list of images already detect
    read_paths.append(new_path)
    img_id = new_path.split('/')[-1].split('.')[0]
    
    # Read the image
    img = cv2.imread(new_path)
    # Getting detections for the image
    detections = detectron.get_detections(img)
    # If the len of detectoints is 0 then, nothing detected
    if len(detections) != 0:
        # Creating image copy - 1 for crop and 2 for polygon
        img2 = img.copy()
        # Sort detections
        detections.sort(reverse=False ,key = lambda x:x[4])
        f.write(new_path + "\n")
        index = 1
        # Loop for each class detected in the image
        # x1, y1, x2, y2 - for the pixels of the polygon
        # cls_conf - the confidence of the model in the class detection
        # cls_pred - class number
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            # Writing to log file the accuracy of current class detected
            f.write("\t+ Label: %s, Conf: %.5f\n" % (classes[int(cls_pred)], cls_conf))        
            
            # Pick the color and font of polygon for the class
            color = colors[int(cls_pred)]
            color = tuple(c*255 for c in color)
            color = (.7*color[2],.7*color[1],.7*color[0])           
            font = cv2.FONT_HERSHEY_SIMPLEX   

            # Polygon pixels
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Text for polygon
            text =  "{} ({}%)".format(classes[int(cls_pred)] ,int(cls_conf * 100))
            
            # Fix pixels of polygon for text input
            cv2.rectangle(img2,(x1,y1) , (x2,y2) , color,3)
            y1 = 0 if y1<0 else y1
            y1_rect = y1-25
            y1_text = y1-5
            if y1_rect<0:
                y1_rect = y1+27
                y1_text = y1+20
            
            # Create the polygon on img2  
            cv2.rectangle(img2,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
            cv2.putText(img2,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            
            # Create the cropped image and save it to the correct folder output  
            new_img = im.open(new_path)
            new_img = new_img.crop((x1,y1, x2, y2))
            new_img.save('{}/Output/Cropped/{}/{}-{}.jpg'.format(output_path, classes[int(cls_pred)],img_id,index))
            index+=1   
            
        f.write("\n")
    else:
        continue
    
    # Print the progress 
    print('done {} images out of {}'.format(image_num, dir_len))
    # Save img2 - the image with the polygons drawen on it
    cv2.imwrite('{}/Output/Full/{}.jpg'.format(output_path, img_id),img2)