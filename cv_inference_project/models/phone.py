import cv2
from ultralytics import YOLO
import logging 

logger = logging.getLogger("inference")

class PhoneModel:
    def __init__(self, conf=0.3):
        logger.debug("Initializing PhoneModel")
        self.model = YOLO("yolov8n.pt")
        self.phone_id = 67
        self.conf = conf

    def predict(self, img):
        res = self.model(img, verbose=False)[0]
        cnt = 0
        for xyxy, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if int(cls)==self.phone_id and conf>self.conf:
                cnt += 1
                x1,y1,x2,y2 = map(int, xyxy)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,165,255),2)
                cv2.putText(img,f"PHONE {conf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
        
        if cnt > 0:
            logger.warning(f"Phone detected: {cnt} times")
        else:
            logger.debug("No phone detected")
        return img, {'phone_count': cnt}

def load_model():
    return PhoneModel()