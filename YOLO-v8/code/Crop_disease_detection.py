import cv2
import numpy as np
from ultralytics import YOLO
import time
from pymavlink import mavutil
import threading
from queue import Queue

class PixhawkSprayController:
    def __init__(self, connection_string='/dev/ttyUSB0', baud=57600):
        """Initialize connection to Pixhawk"""
        try:
            # Connect to Pixhawk using MAVLink
            self.master = mavutil.mavlink_connection(connection_string, baud=baud)
            # Wait for the first heartbeat 
            self.master.wait_heartbeat()
            print("Connected to Pixhawk")
            
            # Initialize spray state
            self.is_spraying = False
            self.spray_command_queue = Queue()
            
        except Exception as e:
            print(f"Error connecting to Pixhawk: {e}")
            raise
    
    def activate_spray(self, channel=6):  # Using AUX6 as default
        """Send command to activate spray system via specified AUX channel"""
        try:
            # Set RC channel to high value (1900) to activate spray
            self.master.mav.rc_channels_override_send(
                self.master.target_system,
                self.master.target_component,
                *[65535]*5,  # Channels 1-5 unchanged
                1900,        # Channel 6 (AUX6) set to 1900 (spray on)
                *[65535]*2   # Channels 7-8 unchanged
            )
            print("Spray system activated")
        except Exception as e:
            print(f"Error activating spray: {e}")

    def deactivate_spray(self, channel=6):  # Using AUX6 as default
        """Send command to deactivate spray system"""
        try:
            # Set RC channel to low value (1100) to deactivate spray
            self.master.mav.rc_channels_override_send(
                self.master.target_system,
                self.master.target_component,
                *[65535]*5,  # Channels 1-5 unchanged
                1100,        # Channel 6 (AUX6) set to 1100 (spray off)
                *[65535]*2   # Channels 7-8 unchanged
            )
            print("Spray system deactivated")
        except Exception as e:
            print(f"Error deactivating spray: {e}")

    def handle_spray_commands(self):
        """Background thread to handle spray commands with timing"""
        while True:
            if not self.spray_command_queue.empty() and not self.is_spraying:
                self.is_spraying = True
                detection = self.spray_command_queue.get()
                
                print(f"Disease detected: {detection['disease']} at confidence: {detection['confidence']}")
                print("Activating spray system for 5 seconds...")
                
                # Activate spray
                self.activate_spray()
                
                # Wait for 5 seconds
                time.sleep(5)
                
                # Deactivate spray
                self.deactivate_spray()
                print("Spray cycle completed")
                
                self.is_spraying = False
            
            time.sleep(0.1)

class DiseaseDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize the YOLO model for disease detection"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def process_frame(self, frame):
        """Process frame and return detections"""
        results = self.model(frame)[0]
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > self.confidence_threshold:
                detections.append({
                    'disease': results.names[int(class_id)],
                    'confidence': score,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
        
        return detections

def main():
    # Initialize Pixhawk controller
    spray_controller = PixhawkSprayController()
    
    # Start spray command handler thread
    spray_thread = threading.Thread(target=spray_controller.handle_spray_commands, daemon=True)
    spray_thread.start()
    
    # Initialize disease detector
    detector = DiseaseDetector('path/to/your/model.pt')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Adjust camera index/path as needed
    
    print("System initialized. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame for disease detection
            detections = detector.process_frame(frame)
            
            # Visual feedback
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{detection['disease']} {detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Queue spray command if disease detected and not currently spraying
                if not spray_controller.is_spraying:
                    spray_controller.spray_command_queue.put(detection)
            
            # Display frame
            cv2.imshow('Disease Detection Feed', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()