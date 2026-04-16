import torch
from PIL import Image
from transformers import pipeline

class VisionProbe:
    def __init__(self):
        print("Initializing Blackwell-Optimized Vision Probe...")
        # device=0 ensures we use the RTX 5070
        self.detector = pipeline(
            model="google/owlvit-base-patch32", 
            task="zero-shot-object-detection", 
            device=0
        )

    def probe_image(self, image_path, candidate_labels):
        """
        Analyzes an image for specific claims/labels.
        """
        image = Image.open(image_path)
        
        # Performance check
        with torch.amp.autocast('cuda'): # Using mixed precision for 50-series speed
            predictions = self.detector(
                image,
                candidate_labels=candidate_labels,
            )
        return predictions

if __name__ == "__main__":
    # Quick test case
    # probe = VisionProbe()
    # print(probe.probe_image("test.jpg", ["laptop", "coffee cup"]))
    pass