import cv2, math
import mediapipe as mp
import logging  # Add this import

logger = logging.getLogger("inference")  # Get the same logger instance

class GazeModel:
    def __init__(self):
        logger.debug("Initializing GazeModel")
        self.mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.idxs = [33, 263, 159, 145]
    
    def _angle(self, p_left, p_right):
        dx, dy = p_right[0]-p_left[0], p_right[1]-p_left[1]
        return math.degrees(math.atan2(dy, dx))
    
    def predict(self, img):
        h, w = img.shape[:2]
        res = self.mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: 
            logger.debug("No face landmarks detected")
            return img, {'gaze_away': False, 'gaze_angle': 0.0}
        
        pts = res.multi_face_landmarks[0].landmark
        p = [(int(pts[i].x*w), int(pts[i].y*h)) for i in self.idxs]
        ang = self._angle(p[0], p[1])
        flag = abs(ang) > 30
        col = (0,0,255) if flag else (0,255,0)
        cv2.arrowedLine(img, p[0], p[1], col, 2)
        cv2.putText(img, f"GazeAway:{flag}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        logger.debug(f"Gaze detected: angle={ang:.1f}°, away={flag}")
        return img, {'gaze_away': flag, 'gaze_angle': ang}

def load_model():
    return GazeModel()