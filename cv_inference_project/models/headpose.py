import cv2, numpy as np, math
import mediapipe as mp
import logging 

logger = logging.getLogger("inference")

class HeadPoseModel:
    def __init__(self):
        logger.debug("Initializing HeadPoseModel")
        self.mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.model_pts = np.array([
            (0.0,   0.0,   0.0), 
            (-30.0, -65.0, -50.0),
            (30.0,  -65.0, -50.0),
            (-40.0, 40.0,  -50.0),
            (40.0,  40.0,  -50.0),
            (0.0,   75.0,  -50.0)
        ])
    
    def _euler(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
        return yaw, pitch, roll

    def predict(self, img):
        h, w = img.shape[:2]
        res = self.mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: 
            logger.debug("No face landmarks detected")
            return img, {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        lm = res.multi_face_landmarks[0].landmark
        image_pts = np.array([
            (lm[1].x*w, lm[1].y*h),
            (lm[33].x*w, lm[33].y*h),
            (lm[263].x*w, lm[263].y*h),
            (lm[61].x*w, lm[61].y*h),
            (lm[291].x*w,lm[291].y*h),
            (lm[199].x*w,lm[199].y*h),
        ], dtype=np.float64)

        cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]])
        _, rvec, _ = cv2.solvePnP(self.model_pts, image_pts, cam, None, flags=0)
        yaw, pitch, roll = self._euler(rvec)
        
        cv2.putText(img,f"Yaw:{yaw:+.1f}", (20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,f"Pitch:{pitch:+.1f}",(20,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,f"Roll:{roll:+.1f}", (20,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
        logger.debug(f"Head pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
        return img, {'yaw': yaw, 'pitch': pitch, 'roll': roll}

def load_model():
    return HeadPoseModel()