import cv2

class AlertSystem:
    def trigger_alert(self, frame, risk_level, risk_color, abnormal_motion=False):
        """
        Draw alerts on the frame if risk is medium/high or abnormal motion detected.
        """
        if risk_level != "NORMAL":
            cv2.putText(frame, f"DENSITY ALERT: {risk_level}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), risk_color, 8)
            
        if abnormal_motion:
            cv2.putText(frame, "MOTION ALERT: ABNORMAL MOVEMENT", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
        return frame
