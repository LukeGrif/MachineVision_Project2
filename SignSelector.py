import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from roidmds import propose_regions
from classifier import SpeedSignClassifier


class SpeedSignDetector:
    def __init__(self, propose_regions_params=None):
        self.classifier = SpeedSignClassifier()
        self.propose_regions_params = propose_regions_params or {}

    def process_image(self, image_path):
        """
        processes the image and returns any speed signs detected

        """
        
        img = np.array(Image.open(image_path))
        if img.shape[2] == 4:  
            img = img[:, :, :3]

        
        regions = propose_regions(img, **self.propose_regions_params, show_plots=True)  

        
        results = []
        for x1, y1, x2, y2 in regions:
            roi = img[y1:y2, x1:x2]
            speed = self.classifier.classify(roi)
            if speed != -1: 
                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'speed': speed
                })

        return results

    def display_results(self, image_path, results):
        """
        Display the input image with bounding boxes and speeds
        """
        img = np.array(Image.open(image_path))
        if img.shape[2] == 4:
            img = img[:, :, :3]

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)

        for detection in results:
            x1, y1, x2, y2 = detection['bbox']
            speed = detection['speed']
          
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            
            ax.text(x1, y1 - 10, f"{speed} km/h",
                    color='red', fontsize=12, weight='bold')

        plt.axis('off')
        plt.show()

    def print_terminal_output(self, results):
        """
        Print detection results to terminal.
        """
        print("\nDetected speed signs:")
        for i, detection in enumerate(results, 1):
            x1, y1, x2, y2 = detection['bbox']
            speed = detection['speed']
            print(f"Sign {i}: {speed} km/h")

        print(f"{len(results)} speed sign{'s' if len(results) != 1 else ''} detected in this image")

import os
import glob
import sys

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python SignSelector.py <image_path>")
        sys.exit(1)

    detector = SpeedSignDetector()
    results = detector.process_image(sys.argv[1])

 
    detector.display_results(sys.argv[1], results)

    
    detector.print_terminal_output(results)