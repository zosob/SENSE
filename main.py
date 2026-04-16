import torch
import cv2
# Import your future SENSE modules here
from src.vision_probe import VisionProbe
# from src.vision_engine import VisionVerify
# from src.symbolic_logic import LogicAudit

def main():
    print("--- SENSE Engine Initializing ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # 1. Receive LLM Claim (Symbolic Input)
    # 2. Process Image (Neural Processing)
    # 3. Verify & Prevent Hallucination
    
    print("System Ready.")

    # Initialize the "Eyes"
    probe = VisionProbe()
    
    # Simulate a "Claim" from an LLM
    llm_claim = ["a high-performance laptop", "a person"]
    image_to_verify = "data/test_ground_truth.jpg"
    
    print(f"SENSE is verifying claim: {llm_claim} against {image_to_verify}")
    
    results = probe.probe_image(image_to_verify, llm_claim)
    
    for res in results:
        print(f"Detected: {res['label']} with {res['score']:.2f} confidence at {res['box']}")

if __name__ == "__main__":
    main()