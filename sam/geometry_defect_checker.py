"""
Bread Geometry Defect Checker - Simple Prototype
Detects basic shape defects: concave areas, bulging, curved edges
"""
import cv2
import numpy as np


class BreadGeometryChecker:
    """Simple bread shape defect detector."""
    
    def __init__(self, 
                 convexity_threshold: float = 0.95,
                 solidity_range: tuple = (0.80, 0.95)):
        self.convexity_threshold = convexity_threshold
        self.solidity_range = solidity_range
    
    def load_mask(self, image_path: str):
        """Load binary mask from image file."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def get_largest_contour(self, binary_mask):
        """Extract the largest contour from binary mask."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
    
    def check_shape_defects(self, contour):
        """Detect both indents (concave) AND bulges (overfilled)."""
        
        # 1. Convexity - catches INDENTS (concave defects)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity_ratio = area / hull_area if hull_area > 0 else 0
        has_indent = convexity_ratio < self.convexity_threshold
        
        # 2. Solidity - catches BULGES (overfilled) and irregular shapes
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        solidity = area / box_area if box_area > 0 else 0
        
        # Normal bread fills 80-95% of bounding box
        # < 80% = irregular/wavy shape
        # > 95% = bulging beyond rectangular shape
        has_bulge = solidity > self.solidity_range[1]
        has_irregular = solidity < self.solidity_range[0]
        
        is_defective = has_indent or has_bulge or has_irregular
        
        return {
            'convexity_ratio': convexity_ratio,
            'solidity': solidity,
            'has_indent': has_indent,
            'has_bulge': has_bulge,
            'has_irregular': has_irregular,
            'is_defective': is_defective
        }
    
    def analyze(self, mask):
        """Analyze bread mask for defects."""
        contour = self.get_largest_contour(mask)
        
        if contour is None or len(contour) < 5:
            return {'error': 'No valid contour', 'is_defective': True}
        
        shape_check = self.check_shape_defects(contour)
        
        defect_types = []
        if shape_check['has_indent']:
            defect_types.append('indent')
        if shape_check['has_bulge']:
            defect_types.append('bulge')
        if shape_check['has_irregular']:
            defect_types.append('irregular')
        
        return {
            'shape': shape_check,
            'is_defective': shape_check['is_defective'],
            'defect_types': defect_types if defect_types else ['normal']
        }
    
    def analyze_from_file(self, image_path: str):
        """Load and analyze image file."""
        mask = self.load_mask(image_path)
        return self.analyze(mask)
    
    def visualize(self, mask, results, save_path=None):
        """Draw defects on the image."""
        contour = self.get_largest_contour(mask)
        
        if contour is None:
            print("No contour found")
            return None
        
        # Create color image
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        shape = results.get('shape', {})
        
        # Draw contour (green for normal, red for defect)
        color = (0, 0, 255) if results['is_defective'] else (0, 255, 0)
        cv2.drawContours(vis, [contour], -1, color, 2)
        
        # Draw convex hull (blue) - shows indents
        hull = cv2.convexHull(contour)
        cv2.drawContours(vis, [hull], -1, (255, 0, 0), 2)
        
        # Draw bounding box (cyan) - shows bulging/irregular
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(vis, [box], -1, (255, 255, 0), 2)
        
        # Add text annotations
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Status
        status_text = "DEFECTIVE" if results['is_defective'] else "NORMAL"
        status_color = (0, 0, 255) if results['is_defective'] else (0, 255, 0)
        cv2.putText(vis, status_text, (10, y_offset), font, font_scale, status_color, thickness)
        y_offset += 30
        
        # Metrics
        cv2.putText(vis, f"Convexity: {shape.get('convexity_ratio', 0):.3f}", 
                   (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(vis, f"Solidity: {shape.get('solidity', 0):.3f}", 
                   (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Defect types
        if results['is_defective']:
            cv2.putText(vis, f"Defects: {', '.join(results['defect_types'])}", 
                       (10, y_offset), font, 0.5, (0, 0, 255), 1)
        
        # Legend
        legend_y = vis.shape[0] - 100
        cv2.putText(vis, "Legend:", (10, legend_y), font, 0.5, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(vis, f"Red/Green = Contour", (10, legend_y), font, 0.4, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(vis, "Blue = Convex Hull", (10, legend_y), font, 0.4, (255, 0, 0), 1)
        legend_y += 20
        cv2.putText(vis, "Cyan = Bounding Box", (10, legend_y), font, 0.4, (255, 255, 0), 1)
        
        if save_path:
            #cv2.imwrite(save_path, vis)
            print(f"Visualization saved to: {save_path}")
        
        return vis


def main():
    """Simple test."""
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "output_frames/binary_mask_000161.jpg"

    checker = BreadGeometryChecker(
        convexity_threshold=0.95,
        solidity_range=(0.80, 0.95)
    )
    
    try:
        results = checker.analyze_from_file(image_path)
        
        print(f"\n{'='*50}")
        print(f"BREAD GEOMETRY ANALYSIS: {image_path}")
        print(f"{'='*50}")
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            return 2
        
        shape = results['shape']
        
        print(f"\nStatus: {'DEFECTIVE ❌' if results['is_defective'] else 'NORMAL ✓'}")
        print(f"\nMetrics:")
        print(f"  Convexity: {shape['convexity_ratio']:.3f} (>0.95 = normal)")
        print(f"  Solidity:  {shape['solidity']:.3f} (0.80-0.95 = normal)")
        
        print(f"\nChecks:")
        print(f"  Indent:    {'❌ YES' if shape['has_indent'] else '✓ NO'}")
        print(f"  Bulge:     {'❌ YES' if shape['has_bulge'] else '✓ NO'}")
        print(f"  Irregular: {'❌ YES' if shape['has_irregular'] else '✓ NO'}")
        
        if results['is_defective']:
            print(f"\nDefect types: {', '.join(results['defect_types'])}")
        
        print(f"{'='*50}\n")
        
        # Create visualization
        mask = checker.load_mask(image_path)
        output_path = image_path.replace('.jpg', '_defects.jpg').replace('.png', '_defects.png')
        vis = checker.visualize(mask, results, save_path=output_path)
        
        # Show image
        if vis is not None:
            cv2.imshow("Bread Defect Analysis", vis)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return 1 if results['is_defective'] else 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit(main())

