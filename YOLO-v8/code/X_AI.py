import torch
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class SugarcaneDiseaseXAI:
    def __init__(self, model_path):
        """
        Initialize YOLO model
        """
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def generate_heatmap(self, image_path):
        """
        Generate a simple heatmap based on model activations
        """
        # Read and preprocess image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        
        # Get YOLO predictions
        results = self.model(image)
        
        # Create heatmap from detection confidence
        heatmap = np.zeros((640, 640))
        
        # For each detection, add to heatmap
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                
                # Convert coordinates to int
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Add confidence score to heatmap region
                heatmap[y1:y2, x1:x2] += conf
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return {
            'original': cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            'heatmap': heatmap,
            'predictions': results
        }
    
    def visualize_results(self, results):
        """
        Visualize the original image and heatmap
        """
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(131)
        plt.imshow(results['original'])
        plt.title('Original Image')
        plt.axis('off')
        
        # Heatmap
        plt.subplot(132)
        plt.imshow(results['heatmap'], cmap='jet')
        plt.title('Detection Heatmap')
        plt.axis('off')
        
        # Superimposed
        plt.subplot(133)
        superimposed = results['original'].copy()
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * results['heatmap']), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(superimposed, 0.7, heatmap_colored, 0.3, 0)
        plt.imshow(superimposed)
        plt.title('Superimposed')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_predictions(self, results):
        """
        Analyze the model's predictions
        """
        if results['predictions'][0].boxes is not None:
            boxes = results['predictions'][0].boxes
            analysis = {
                'num_detections': len(boxes),
                'confidence_scores': [conf.item() for conf in boxes.conf],
                'average_confidence': boxes.conf.mean().item(),
                'classes_detected': boxes.cls.cpu().numpy().tolist()
            }
        else:
            analysis = {
                'num_detections': 0,
                'confidence_scores': [],
                'average_confidence': 0,
                'classes_detected': []
            }
        return analysis

def main():
    # Initialize the XAI system
    model_path = "YOLOv8_sugarcane.pt"  # Replace with your model path
    xai_system = SugarcaneDiseaseXAI(model_path)
    
    # Analyze an image
    image_path = "test.jpeg"  # Replace with your image path
    
    # Generate results
    results = xai_system.generate_heatmap(image_path)
    
    # Visualize results
    xai_system.visualize_results(results)
    
    # Get analysis
    analysis = xai_system.analyze_predictions(results)
    print("\nAnalysis Results:")
    print(f"Number of detections: {analysis['num_detections']}")
    print(f"Average confidence: {analysis['average_confidence']:.2f}")
    print(f"Classes detected: {analysis['classes_detected']}")

if __name__ == "__main__":
    main()