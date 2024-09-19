import sys 
import os.path
import random
import cv2
import numpy as np
import torch
import json
from PIL import Image
#faarom pre_image import letterbox
#from pre_image import pre_precess_image
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
# names=["crane","collapsed_crane", "tower_crane",  "drilling_rig", "concrete_mixing_truck", "excavating_machinery", "bulldozer","roller"]
sys.path.insert(1,'/project/train/src_repo/')
#model_name = '/project/train/models/train3/weights/best.pt'
weights = r'/project/train/models/train/weights/best.pt'


def init():
    global cuda, conf_thres, iou_thres,names,imgsz,half    
    cuda = 'cuda:0'
    conf_thres = 0.15
    iou_thres = 0.45
    half = False
    imgsz = 640
    weights = '/project/train/models/train/weights/best.pt'
    names = ['crane', 'collapsed_crane', 'tower_crane', 'drilling_rig', 'concrete_mixing_truck', 'excavating_machinery', 
'bulldozer', 'roller', 'other_large_lifting_equipment', 'mechanical_arm', 'mechanical_base']
    model = AutoBackend(weights, device=torch.device(cuda))
    model.eval()    
    return model
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    #print(im.shape)
    # shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / im.shape[0], new_shape[1] / im.shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(im.shape[1] * r)), int(round(im.shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if im.shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
def pre_precess_image(img_src, img_size, half, device):
        # Padded resize
    img = letterbox(img_src, img_size)[0]
    #print(img.shape)
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img
def process_image(weights, input_image,args=None, **kwargs):
    #print('***********************',input_image)
    img_src = input_image #cv2.imread(input_image)
    shape_src = input_image.shape
    img = pre_precess_image(img_src, imgsz, half, cuda)
    names = ['crane', 'collapsed_crane', 'tower_crane', 'drilling_rig', 'concrete_mixing_truck', 'excavating_machinery', 
'bulldozer', 'roller', 'other_large_lifting_equipment', 'mechanical_arm', 'mechanical_base']
    preds = weights(img)
    det = ops.non_max_suppression(preds, conf_thres, iou_thres, classes=None, agnostic=False, max_det=300, nc=len(names))
    fake_result = {}
    #print('********************',det)

    fake_result["algorithm_data"] = {
       "is_alert": 'false',
       "target_count": 0,
       "target_info": [],
       #"model_data": {"objects": []} 
    }
    cnt = 0
    fake_result["model_data"] = {"objects": []}
    for i, pred in enumerate(det):
        #lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
        #tf = max(lw - 1, 1)  # font thickness
        #sf = lw / 3  # font scale
        # try:
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape_src)
        results = pred.cpu().detach().numpy()
        for result in results:
            #print(result[5])
            #print(result)
            xmin = int(result[0])
            ymin = int(result[1])
            xmax = int(result[2])
            ymax = int(result[3])

            names = names
            # print(xmin,ymin,xmax,ymax,names[int(det.boxes.cls)])    
            #fake_result["algorithm_data"]["model_data"]['objects'].append({
            fake_result["model_data"]["objects"].append({
                "x":xmin,
                "y":ymin,
                "width":xmax-xmin,
                "height":ymax-ymin,
                "confidence":float(result[4]),
                "name":names[int(result[5])]
            })
            cnt+=1
            fake_result["algorithm_data"]["target_info"].append({
                "x":xmin,
                "y":ymin,
                "width":xmax-xmin,
                "height":ymax-ymin,
                "confidence":float(result[4]),
                "name":names[int(result[5])]
            })
            fake_result["algorithm_data"]["target_count"] += 1
        # except:
        #     pass
    if cnt>0:
        fake_result["algorithm_data"]["is_alert"] = 'true'
        fake_result["algorithm_data"]["target_count"] = cnt
    else:
        fake_result["algorithm_data"]["target_info"]=[]

    # fake_result["algorithm_data"]["target_info"]=[]
    #return fake_result
    return json.dumps(fake_result, indent = 4)



import time

if __name__ == '__main__':
    #save_path = "/home/yang/xiaoma/code_new/ultralytics-1/runs"
    from glob import glob
    # Test API
    files = []
    file = glob('/home/data/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/*.jpg')
    for i in file:
        files.append(i)

    s = 0
    weights = init()
    for img_path in files:

        t1 = time.time()
        #print(save_path)
        # process_image = YOLOV8DetectionInfer(weights, cuda, conf_thres, iou_thres)
        res = process_image(weights,img_path)
        #print(res)
        t2 = time.time()
        s += t2 - t1
        # break
    #print(1/(s/100))

