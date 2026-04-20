class CrowdDensityAnalyzer:
    def __init__(self, medium_risk_threshold=10, high_risk_threshold=20):
        """
        Initialize thresholds for crowd density.
        Values are number of people detected in the frame.
        """
        self.medium_risk_threshold = medium_risk_threshold
        self.high_risk_threshold = high_risk_threshold

    def analyze(self, num_people):
        """
        Analyze the crowd density.
        Returns risk level and a color for displaying the alert.
        """
        if num_people >= self.high_risk_threshold:
            return "HIGH RISK", (0, 0, 255) # Red
        elif num_people >= self.medium_risk_threshold:
            return "MEDIUM RISK", (0, 165, 255) # Orange
        else:
            return "NORMAL", (0, 255, 0) # Green
