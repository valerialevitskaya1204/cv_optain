import cv2
import numpy as np
from insightface.app import FaceAnalysis
import logging

logger = logging.getLogger("inference")

class IdentityModel:
    def __init__(self, thr: float = 1.0):
        logger.debug("Initializing IdentityModel")
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.ref_vec = None
        self.thr = thr

    def _get_vec(self, img):
        faces = self.app.get(img)
        return faces[0].embedding if faces else None

    def predict(self, img):
        if self.ref_vec is None:
            vec = self._get_vec(img)
            if vec is None:
                logger.warning("No face detected in enrollment frame")
                cv2.putText(img, "NO FACE!", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                return img, {'is_match': False, 'distance': float('inf')}
            else:
                self.ref_vec = vec
                logger.info("Identity enrolled successfully")
                return img, {'is_match': True, 'distance': 0.0}
        
        cur = self._get_vec(img)
        if cur is None:
            cv2.putText(img, "NO FACE!", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            return img, {'is_match': False, 'distance': float('inf')}
        
        dist = np.linalg.norm(self.ref_vec - cur)
        ok = dist < self.thr
        text = f"{'MATCH' if ok else 'IMPOSTOR'} {dist:.2f}"
        color = (0,255,0) if ok else (0,0,255)
        cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        return img, {'is_match': ok, 'distance': dist}

def load_model():
    return IdentityModel()