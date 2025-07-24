import cv2
from ultralytics import YOLO
import logging 

logger = logging.getLogger("inference")

class PersonsModel:
    def __init__(self, conf=0.25):
        logger.debug("Initializing PersonsModel")
        self.model = YOLO("yolov8n.pt")
        self.person_id = 0
        self.conf = conf

    def predict(self, img):
        res = self.model(img, verbose=False)[0]
        cnt = sum(int(c)==self.person_id and cf>self.conf
                for c,cf in zip(res.boxes.cls, res.boxes.conf))
        flag = cnt > 1
        col = (0,0,255) if flag else (0,255,0)
        cv2.putText(img,f"Persons:{cnt}",(20,190),
                    cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
        
        if flag:
            logger.warning(f"Multiple persons detected: {cnt}")
        else:
            logger.debug(f"Person count: {cnt}")
        return img, {'person_count': cnt}

def load_model():
    return PersonsModel()