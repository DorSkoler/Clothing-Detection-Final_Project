from numpy import append
import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
from tqdm import tqdm
from PIL import Image as im

output_path = '../'
input_path = '../Test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

yolo_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.7,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#Classes
classes = load_classes(yolo_params["class_path"])

#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])


detectron = YOLOv3Predictor(params=yolo_params)

    
all_dir = ["", "/Cropped", "/Delete", "/Full"]

for dir in all_dir:
    if not os.path.isdir('{}/Output{}'.format(output_path, dir)):
        os.mkdir('{}/Output{}'.format(output_path, dir)) 

for cls in  classes:
    if os.path.isdir('{}/Output/Cropped/{}'.format(output_path, cls)):
        continue
    os.mkdir('{}/Output/Cropped/{}'.format(output_path, cls))
    
dir_len = len(os.listdir(input_path))
image_num = 0
read_paths = []
f = open('{}/Output/accuracy.txt'.format(output_path), 'a')
while(dir_len != image_num):

    for img_path in os.listdir(input_path):
        image_num+=1
        new_path = '{}/{}'.format(input_path, img_path)
        
        if new_path in read_paths:
            continue
    
        read_paths.append(new_path)
        img = cv2.imread(new_path)
        detections = detectron.get_detections(img)
        if len(detections) != 0:
            img2 = img.copy()
            detections.sort(reverse=False ,key = lambda x:x[4])
            f.write(new_path + "\n")
            for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                    
                f.write("\t+ Label: %s, Conf: %.5f\n" % (classes[int(cls_pred)], cls_conf))        
                color = colors[int(cls_pred)]
                
                color = tuple(c*255 for c in color)
                color = (.7*color[2],.7*color[1],.7*color[0])       
                    
                font = cv2.FONT_HERSHEY_SIMPLEX   
            
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                
                cv2.rectangle(img2,(x1,y1) , (x2,y2) , color,3)
                y1 = 0 if y1<0 else y1
                y1_rect = y1-25
                y1_text = y1-5
                if y1_rect<0:
                    y1_rect = y1+27
                    y1_text = y1+20
                    
                cv2.rectangle(img2,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
                cv2.putText(img2,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                    
                img_id = new_path.split('/')[-1].split('.')[0]
                cv2.imwrite('{}/Output/Delete/{}.jpg'.format(output_path, img_id),img)
                new_img = im.open('{}/Output/Delete/{}.jpg'.format(output_path, img_id))
                new_img = new_img.crop((x1,y1, x2, y2))
                new_img.save('{}/Output/Cropped/{}/{}.jpg'.format(output_path, classes[int(cls_pred)],img_id))    
                
            f.write("\n")
        else:
            continue
            
        print('done {} images out of {}'.format(image_num, dir_len))
        img_id = new_path.split('/')[-1].split('.')[0]
        cv2.imwrite('{}/Output/Full/{}.jpg'.format(output_path, img_id),img2)
        

for f in os.listdir('{}/Output/Delete'.format(output_path)):
    os.remove('{}/Output/Delete/{}'.format(output_path, f))
os.rmdir('{}/Output/Delete'.format(output_path))