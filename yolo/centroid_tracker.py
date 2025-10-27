import math

class SimpleBreadTracker:
    """Super simple tracker - just tracks center points of bread"""
    
    def __init__(self):
        self.next_id = 0
        self.tracked_breads = {}  # {id: {'center': (x,y), 'missing_frames': 0, 'box': (x1,y1,x2,y2), 'crack_percentage': 0.0, 'is_defective': False}}
        self.max_missing_frames = 10  # Remove bread after 10 missing frames
        self.max_distance = 150  # Max pixels bread can move between frames
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections = [(x1, y1, x2, y2), (x1, y1, x2, y2), ...] - list of bounding boxes
        """
        # Calculate centers of new detections
        new_centers = []
        for x1, y1, x2, y2 in detections:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            new_centers.append((center_x, center_y))
        
        # If no existing breads, just register all new ones
        if len(self.tracked_breads) == 0:
            for i, (center, box) in enumerate(zip(new_centers, detections)):
                self.tracked_breads[self.next_id] = {
                    'center': center,
                    'missing_frames': 0,
                    'box': box,
                    'crack_percentage': 0.0,
                    'is_defective': False,
                    'num_cracks': 0
                }
                self.next_id += 1
            return self.tracked_breads
        
        # Match new detections to existing breads
        used_detections = set()
        updated_breads = set()
        
        # For each existing bread, find closest new detection
        for bread_id, bread_data in list(self.tracked_breads.items()):
            old_center = bread_data['center']
            best_distance = float('inf')
            best_match = None
            
            for i, new_center in enumerate(new_centers):
                if i in used_detections:  # Already matched
                    continue
                
                distance = self.calculate_distance(old_center, new_center)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update existing bread
                self.tracked_breads[bread_id]['center'] = new_centers[best_match]
                self.tracked_breads[bread_id]['box'] = detections[best_match]
                self.tracked_breads[bread_id]['missing_frames'] = 0
                used_detections.add(best_match)
                updated_breads.add(bread_id)
            else:
                # Bread not found, increment missing counter
                self.tracked_breads[bread_id]['missing_frames'] += 1
        
        # Add new breads for unmatched detections
        for i, (center, box) in enumerate(zip(new_centers, detections)):
            if i not in used_detections:
                self.tracked_breads[self.next_id] = {
                    'center': center,
                    'missing_frames': 0,
                    'box': box,
                    'crack_percentage': 0.0,
                    'is_defective': False,
                    'num_cracks': 0
                }
                self.next_id += 1
        
        # Remove breads that have been missing too long
        to_remove = []
        for bread_id, bread_data in self.tracked_breads.items():
            if bread_data['missing_frames'] > self.max_missing_frames:
                to_remove.append(bread_id)
        
        for bread_id in to_remove:
            del self.tracked_breads[bread_id]
        
        return self.tracked_breads

