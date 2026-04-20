import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self, width, height):
        """
        Initialize the heatmap generator with frame dimensions.
        """
        self.accum_image = np.zeros((height, width), dtype=np.float32)

    def update(self, boxes):
        """
        Update the heatmap accumulater with current bounding boxes.
        """
        current_heat = np.zeros_like(self.accum_image)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Draw a circle for each person detected
            cv2.circle(current_heat, (cx, cy), radius=40, color=1, thickness=-1)
            
        # Accumulate with decay to keep memory of past frames
        self.accum_image = cv2.addWeighted(self.accum_image, 0.99, current_heat, 0.05, 0)
        
    def get_heatmap_overlay(self, frame):
        """
        Return the original frame with the heatmap overlaid.
        """
        heat_norm = cv2.normalize(self.accum_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        
        # Create a mask where heat is present
        mask = heat_norm > 20
        
        result = frame.copy()
        overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
        result[mask] = overlay[mask]
        return result
