"""
Defect Analyzer Module
Handles crack detection, shape matching, and defect classification
"""
import cv2
import numpy as np


class DefectAnalyzer:
    """Analyzes bread for defects using crack detection and shape matching"""
    
    def __init__(self, template_contour, shape_threshold=0.15, crack_threshold=5.0, iou_threshold=0.3):
        """
        Initialize defect analyzer
        
        Args:
            template_contour: Reference contour for shape matching
            shape_threshold: Threshold for shape matching score
            crack_threshold: Threshold for crack percentage
            iou_threshold: IoU threshold for crack-bread association
        """
        self.template_contour = template_contour
        self.shape_threshold = shape_threshold
        self.crack_threshold = crack_threshold
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
        
        Returns:
            float: IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_crack_in_bread(self, crack_box, bread_box):
        """
        Check if crack bounding box belongs to bread bounding box
        
        Args:
            crack_box: (x1, y1, x2, y2) of crack
            bread_box: (x1, y1, x2, y2) of bread
        
        Returns:
            bool: True if crack belongs to bread
        """
        iou = self.calculate_iou(crack_box, bread_box)
        
        # Also check if crack center is within bread box
        crack_center_x = (crack_box[0] + crack_box[2]) / 2
        crack_center_y = (crack_box[1] + crack_box[3]) / 2
        
        x1, y1, x2, y2 = bread_box
        center_in_bread = (x1 <= crack_center_x <= x2) and (y1 <= crack_center_y <= y2)
        
        return iou > self.iou_threshold or center_in_bread
    
    def analyze_bread(self, bread_box, bread_mask, crack_detections, crack_masks, 
                     shape_threshold=None, crack_threshold=None):
        """
        Analyze a single bread for defects
        
        Args:
            bread_box: (x1, y1, x2, y2) bounding box
            bread_mask: Binary mask of the bread
            crack_detections: List of crack bounding boxes
            crack_masks: List of crack masks
            shape_threshold: Override default shape threshold
            crack_threshold: Override default crack threshold
        
        Returns:
            dict: Analysis results containing:
                - crack_percentage: float
                - match_score: float
                - shape_defective: bool
                - crack_defective: bool
                - is_defective: bool
                - num_cracks: int
                - associated_cracks: list of crack indices
        """
        shape_thresh = shape_threshold if shape_threshold is not None else self.shape_threshold
        crack_thresh = crack_threshold if crack_threshold is not None else self.crack_threshold
        
        results = {
            'crack_percentage': 0.0,
            'match_score': 0.0,
            'shape_defective': False,
            'crack_defective': False,
            'is_defective': False,
            'num_cracks': 0,
            'associated_cracks': []
        }
        
        if bread_mask is None:
            return results
        
        # Calculate bread area from mask and extract contour for shape matching
        bread_binary_mask = (bread_mask > 0.5).astype(np.uint8)
        bread_area = np.sum(bread_binary_mask)
        
        # Get contour for shape matching
        bread_binary_mask_255 = (bread_binary_mask * 255).astype(np.uint8)
        bread_contours, _ = cv2.findContours(bread_binary_mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bread_contour = max(bread_contours, key=cv2.contourArea) if bread_contours else None
        
        # Perform shape matching with template
        if bread_contour is not None and len(bread_contour) > 5:
            match_score = cv2.matchShapes(self.template_contour, bread_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            shape_defective = match_score > shape_thresh
            results['match_score'] = match_score
            results['shape_defective'] = shape_defective
        
        # Find all cracks that belong to this bread
        total_crack_area = 0
        associated_cracks = []
        
        for crack_idx, crack_box in enumerate(crack_detections):
            if self.is_crack_in_bread(crack_box, bread_box):
                associated_cracks.append(crack_idx)
                if crack_idx < len(crack_masks) and crack_masks[crack_idx] is not None:
                    crack_binary_mask = (crack_masks[crack_idx] > 0.5).astype(np.uint8)
                    crack_area = np.sum(crack_binary_mask)
                    total_crack_area += crack_area
        
        # Calculate crack percentage
        if bread_area > 0:
            crack_percentage = (total_crack_area / bread_area) * 100.0
        else:
            crack_percentage = 0.0
        
        crack_defective = crack_percentage > crack_thresh
        
        # Combined defect detection: defective if EITHER shape is bad OR cracks exceed threshold
        results['crack_percentage'] = crack_percentage
        results['crack_defective'] = crack_defective
        results['is_defective'] = results['shape_defective'] or crack_defective
        results['num_cracks'] = len(associated_cracks)
        results['associated_cracks'] = associated_cracks
        
        return results
    
    def update_thresholds(self, shape_threshold=None, crack_threshold=None, iou_threshold=None):
        """Update detection thresholds"""
        if shape_threshold is not None:
            self.shape_threshold = shape_threshold
        if crack_threshold is not None:
            self.crack_threshold = crack_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
