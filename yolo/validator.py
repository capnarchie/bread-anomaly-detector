import time 
import cv2

class SimpleValidator:
    def __init__(self):
        self.required_frames = 3
        self.window_size = 5
        self.detections = []  # List of recent detections
        self.exit_detections = []  # List of recent exit detections

    def add_detection(self, is_defective, bread_id):
        self.detections.append((is_defective, bread_id))
        if len(self.detections) > self.window_size:
            self.detections.pop(0)

    def is_valid(self, bread_id):
        defect_count = 0
        for i in self.detections:
            if i[1] == bread_id:
                if i[0] == True:
                    defect_count += 1
        if defect_count >= self.required_frames:
            #self.is_defective = True
            return True
        else:
            #self.is_defective = False
            return False
    
    # Keep track if centroid is in validation zone
    def is_centroid_in_validation_zone(self, contour, validation_zone):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x_min, y_min, x_max, y_max = validation_zone
        return x_min <= cX <= x_max and y_min <= cY <= y_max

    def is_centroid_in_trigger_zone(self, contour, trigger_zone):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x_min, y_min, x_max, y_max = trigger_zone
        return x_min <= cX <= x_max and y_min <= cY <= y_max

    def add_exit_detection(self, bread_id):
        self.exit_detections.append(bread_id)
        if len(self.exit_detections) >= 2:
            self.exit_detections.pop(0)

    def can_exit_detection(self, bread_id):
        return True if bread_id in self.exit_detections else False