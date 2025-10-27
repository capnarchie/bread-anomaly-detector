"""
UI Renderer Module
Handles all visualization and drawing operations
"""
import cv2


class UIRenderer:
    """Renders UI elements for bread tracking visualization"""
    
    def __init__(self, window_name='Simple Bread Tracking'):
        """
        Initialize UI renderer
        
        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        
        # Colors
        self.COLOR_GOOD = (0, 255, 0)      # Green
        self.COLOR_BAD = (0, 0, 255)       # Red
        self.COLOR_MISSING = (0, 255, 255) # Yellow
        self.COLOR_CRACK = (255, 0, 255)   # Magenta
        self.COLOR_ZONE = (255, 0, 0)      # Blue
        self.COLOR_ZONE_TEXT = (0, 255, 255) # Cyan
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
    
    def draw_bread_tracking(self, frame, tracked_breads):
        """
        Draw bounding boxes and labels for tracked breads
        
        Args:
            frame: Image frame to draw on
            tracked_breads: Dictionary of tracked bread data
        """
        for bread_id, bread_data in tracked_breads.items():
            x1, y1, x2, y2 = bread_data['box']
            center = bread_data['center']
            missing = bread_data['missing_frames']
            crack_percentage = bread_data.get('crack_percentage', 0.0)
            match_score = bread_data.get('match_score', 0.0)
            shape_defective = bread_data.get('shape_defective', False)
            crack_defective = bread_data.get('crack_defective', False)
            is_defective = bread_data.get('is_defective', False)
            
            # Choose color based on status
            if is_defective:
                color = self.COLOR_BAD
            elif missing > 0:
                color = self.COLOR_MISSING
            else:
                color = self.COLOR_GOOD
            
            # Draw bounding box and center point
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 5, color, -1)
            
            # Build label
            label = f"Bread_{bread_id}"
            if missing > 0:
                label += f" (Missing: {missing})"
            
            # Build detailed status
            defect_reasons = []
            if shape_defective:
                defect_reasons.append(f"Shape:{match_score:.2f}")
            if crack_defective:
                defect_reasons.append(f"Crack:{crack_percentage:.1f}%")
            
            if defect_reasons:
                label += f" | {' + '.join(defect_reasons)}"
            elif crack_percentage > 0 or match_score > 0:
                label += f" | S:{match_score:.2f} C:{crack_percentage:.1f}%"
            
            if is_defective:
                label += " [REJECT]"
            
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def draw_cracks(self, frame, crack_detections):
        """
        Draw crack bounding boxes
        
        Args:
            frame: Image frame to draw on
            crack_detections: List of crack bounding boxes
        """
        for crack_box in crack_detections:
            x1, y1, x2, y2 = crack_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_CRACK, 1)
    
    def draw_validation_info(self, frame, bread_data, is_valid):
        """
        Draw validation information on bread in validation zone
        
        Args:
            frame: Image frame to draw on
            bread_data: Dictionary containing bread information
            is_valid: Boolean indicating if bread is validated as defective
        """
        x1, y1, x2, y2 = bread_data['box']
        crack_percentage = bread_data.get('crack_percentage', 0.0)
        match_score = bread_data.get('match_score', 0.0)
        shape_defective = bread_data.get('shape_defective', False)
        crack_defective = bread_data.get('crack_defective', False)
        
        status_text = "BAD" if is_valid else "OK"
        color = self.COLOR_BAD if is_valid else self.COLOR_GOOD
        
        cv2.putText(frame, f"Validation: {status_text}", (x1, y2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Shape: {match_score:.3f} {'BAD' if shape_defective else 'OK'}", 
                   (x1, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Crack: {crack_percentage:.1f}% {'BAD' if crack_defective else 'OK'}", 
                   (x1, y2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def draw_reject_label(self, frame, bread_data):
        """
        Draw reject label on bread in exit zone
        
        Args:
            frame: Image frame to draw on
            bread_data: Dictionary containing bread information
        """
        x1, y1, x2, y2 = bread_data['box']
        cv2.putText(frame, "REJECT", (x1, y2 + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, self.COLOR_BAD, 3)
    
    def draw_info_panel(self, frame, stats):
        """
        Draw information panel with statistics
        
        Args:
            frame: Image frame to draw on
            stats: Dictionary containing statistics to display
                - fps: float
                - total_count: int
                - defective_count: int
                - shape_defective: int
                - crack_defective: int
                - shape_threshold: float
                - crack_threshold: float
                - total_cracks: int
        """
        # Draw background
        cv2.rectangle(frame, (10, 10), (550, 170), self.COLOR_BLACK, -1)
        
        # Draw text
        cv2.putText(frame, f"FPS: {stats.get('fps', 0):.1f}", 
                   (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_WHITE, 2)
        cv2.putText(frame, f"Breads: {stats.get('total_count', 0)} | Defective: {stats.get('defective_count', 0)}", 
                   (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_WHITE, 2)
        cv2.putText(frame, f"Shape Threshold: {stats.get('shape_threshold', 0):.2f} (Bad: {stats.get('shape_defective', 0)})", 
                   (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WHITE, 2)
        cv2.putText(frame, f"Crack Threshold: {stats.get('crack_threshold', 0)}% (Bad: {stats.get('crack_defective', 0)})", 
                   (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WHITE, 2)
        cv2.putText(frame, f"Total Cracks: {stats.get('total_cracks', 0)}", 
                   (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WHITE, 2)
    
    def draw_zones(self, frame, validation_zone, exit_zone):
        """
        Draw validation and exit zones
        
        Args:
            frame: Image frame to draw on
            validation_zone: (x1, y1, x2, y2) validation zone coordinates
            exit_zone: (x1, y1, x2, y2) exit zone coordinates
        """
        # Validation zone
        x1_val, y1_val, x2_val, y2_val = validation_zone
        cv2.rectangle(frame, (x1_val, y1_val), (x2_val, y2_val), self.COLOR_ZONE, 2)
        cv2.putText(frame, "Validation", (x1_val, y1_val - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_ZONE_TEXT, 2)
        
        # Exit zone
        x1_exit, y1_exit, x2_exit, y2_exit = exit_zone
        cv2.rectangle(frame, (x1_exit, y1_exit), (x2_exit, y2_exit), self.COLOR_ZONE, 2)
        cv2.putText(frame, "Reject", (x1_exit, y1_exit - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_ZONE_TEXT, 2)
    
    def show_frame(self, frame):
        """Display the frame"""
        cv2.imshow(self.window_name, frame)
    
    @staticmethod
    def create_window(window_name, width=800, height=600):
        """
        Create and configure OpenCV window
        
        Args:
            window_name: Name of the window
            width: Window width
            height: Window height
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
    
    @staticmethod
    def create_trackbar(name, window_name, default_value, max_value, callback):
        """
        Create a trackbar
        
        Args:
            name: Trackbar name
            window_name: Window to attach trackbar to
            default_value: Initial value
            max_value: Maximum value
            callback: Callback function
        """
        cv2.createTrackbar(name, window_name, default_value, max_value, callback)
    
    @staticmethod
    def get_trackbar_value(name, window_name):
        """Get current trackbar value"""
        return cv2.getTrackbarPos(name, window_name)
