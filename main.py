import torch
import cv2
# Import your future SENSE modules here
from src.vision_probe import VisionProbe
from src.logic_gate import LogicGate
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
    gate = LogicGate(threshold=0.75) # Set your "Truth Bar"
    
    # Simulate a "Claim" from an LLM
    llm_claim = ["a high-performance laptop", "a person", "a plane", "a ball", "a plant", "a card"]
    image_to_verify = "data/test_ground_truth1.jpg"
    
    print(f"SENSE is verifying claim: {llm_claim} against {image_to_verify}")
    
    results = probe.probe_image(image_to_verify, llm_claim)
    final_report = gate.audit(llm_claim, results)
    
    for res in results:
        print(f"Detected: {res['label']} with {res['score']:.2f} confidence at {res['box']}")

    print("\n--- SENSE Audit Report ---")
    for item in final_report:
        print(f"[{item['status']}] Claim: {item['claim']}")

if __name__ == "__main__":
    main()