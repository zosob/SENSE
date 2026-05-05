import torch
import cv2
# Import your future SENSE modules here
from src.vision_probe import VisionProbe
from src.logic_gate import LogicGate
import time
from src.visualizer import Visualizer
from src.temporal_tracker import TemporalTracker
# from src.vision_engine import VisionVerify
# from src.symbolic_logic import LogicAudit

def main():
    
    print("--- SENSE Engine Initializing ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # 1. Receive LLM Claim (Symbolic Input)
    # 2. Process Image (Neural Processing)
    # 3. Verify & Prevent Hallucination
    
    print("System Ready.")

    # 1. Initialize the "Eyes"... the three pillars
    probe = VisionProbe()
    gate = LogicGate(threshold=0.3) # Set your "Truth Bar"
    tracker = TemporalTracker(patience=5) #holds objects in memory for 5 frames
    viz = Visualizer()

    # Defining the "Claim" from an LLM
    llm_claim = ["laptop", "person", "red vase", "picture", "ball", "tree", "card", "plant on vase", "vase on desk", "mouse", "book", "desk"]
    image_to_verify = "data/test_ground_truth1.jpg"
    
    video_source = "data/test_video.mp4" # Or use 0 for webcam
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        print(f"SENSE is verifying claim: {llm_claim} against {image_to_verify}")
        
        results = probe.probe_batch([frame], llm_claim)
        
        # Temporal Step
        tracker.update(results)
        buffered_results = tracker.get_buffered_detections(results)
        
        # Symbolic Step
        final_report = gate.audit(llm_claim, buffered_results)
        
        # Visual Step
        viz.draw_results(frame, buffered_results, final_report, is_video = True)
        
        # Display FPS    
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"SENSE FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("SENSE Temporal Audit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Audit Complete.")

    print(f"SENSE Inference Speed: {fps} FPS")

if __name__ == "__main__":
    main()