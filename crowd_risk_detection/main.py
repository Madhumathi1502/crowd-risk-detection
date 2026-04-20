import cv2
import argparse
from camera import CameraStream
from detection import PersonDetector
from density import CrowdDensityAnalyzer
from heatmap import HeatmapGenerator
from motion import MotionAnalyzer
from alert import AlertSystem

def main():
    parser = argparse.ArgumentParser(description="Crowd Risk Detection System")
    parser.add_argument('--source', default=1, help='Camera source index or DroidCam IP URL')
    args = parser.parse_args()
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Initialize Modules
    cam = CameraStream(source)
    if not cam.is_opened():
        print(f"Error: Cannot open camera source {source}")
        return

    detector = PersonDetector(model_path='yolov8n.pt')
    density_analyzer = CrowdDensityAnalyzer(medium_risk_threshold=10, high_risk_threshold=20)
    motion_analyzer = MotionAnalyzer()
    alert_system = AlertSystem()
    
    heatmap_gen = None

    print("Starting Crowd Risk Detection System...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cam.get_frame()
        if not ret or frame is None:
            print("Failed to read frame or stream ended.")
            break
        if heatmap_gen is None:
            height, width = frame.shape[:2]
            heatmap_gen = HeatmapGenerator(width=width, height=height)

        # 1. Detection
        annotated_frame, boxes = detector.process_frame(frame)
        num_people = len(boxes)

        # 2. Crowd Density Analysis
        risk_level, risk_color = density_analyzer.analyze(num_people)

        # 3. Motion Analysis (abnormal movement)
        abnormal_motion = motion_analyzer.analyze(boxes)

        # 4. Heatmap Update and Overlay
        heatmap_gen.update(boxes)
        # Apply heatmap on top of the original annotated frame
        heatmap_frame = heatmap_gen.get_heatmap_overlay(annotated_frame)

        # 5. Alert System
        final_frame = alert_system.trigger_alert(heatmap_frame, risk_level, risk_color, abnormal_motion)

        # Display people count
        cv2.putText(final_frame, f"People Count: {num_people}", (30, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show Output
        cv2.imshow("Crowd Risk Detection System", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
