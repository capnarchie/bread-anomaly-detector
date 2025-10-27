"""
Zone Manager Module
Handles validation and exit zone logic
"""


class ZoneManager:
    """Manages validation zones and exit zones for bread inspection"""
    
    def __init__(self, frame_width, frame_height):
        """
        Initialize zone manager
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Default zone parameters (can be overridden)
        self.validation_zone_width_percent = 0.1  # 10% of frame width
        self.validation_zone_height_percent = 0.35  # 35% of frame height
        
        # Fixed exit zone (can be made configurable)
        self.exit_zone = (900, 175, 1200, 900)  # (x1, y1, x2, y2)
        
        self._update_validation_zone()
    
    def _update_validation_zone(self):
        """Calculate validation zone based on frame center"""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        
        x1 = int(center_x - self.validation_zone_width_percent * self.frame_width)
        x2 = int(center_x + self.validation_zone_width_percent * self.frame_width)
        y1 = int(center_y - self.validation_zone_height_percent * self.frame_height)
        y2 = int(center_y + self.validation_zone_height_percent * self.frame_height)
        
        self.validation_zone = (x1, y1, x2, y2)
    
    def update_frame_size(self, width, height):
        """Update frame dimensions and recalculate zones"""
        self.frame_width = width
        self.frame_height = height
        self._update_validation_zone()
    
    def is_in_validation_zone(self, point):
        """
        Check if a point is within the validation zone
        
        Args:
            point: (x, y) coordinates
        
        Returns:
            bool: True if point is in validation zone
        """
        x, y = point
        x1, y1, x2, y2 = self.validation_zone
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    
    def is_in_exit_zone(self, point):
        """
        Check if a point is within the exit zone
        
        Args:
            point: (x, y) coordinates
        
        Returns:
            bool: True if point is in exit zone
        """
        x, y = point
        x1, y1, x2, y2 = self.exit_zone
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    
    def get_validation_zone(self):
        """Get validation zone coordinates (x1, y1, x2, y2)"""
        return self.validation_zone
    
    def get_exit_zone(self):
        """Get exit zone coordinates (x1, y1, x2, y2)"""
        return self.exit_zone
    
    def set_validation_zone_size(self, width_percent, height_percent):
        """
        Set validation zone size as percentage of frame
        
        Args:
            width_percent: Width as fraction (e.g., 0.1 for 10%)
            height_percent: Height as fraction (e.g., 0.35 for 35%)
        """
        self.validation_zone_width_percent = width_percent
        self.validation_zone_height_percent = height_percent
        self._update_validation_zone()
    
    def set_exit_zone(self, x1, y1, x2, y2):
        """Set custom exit zone coordinates"""
        self.exit_zone = (x1, y1, x2, y2)
