class TemporalTracker:
    def __init__(self, patience=5):
        # How many frames SENSE will "remember" an unseen object
        self.patience = patience 
        # Stores {label: {"box": coordinates, "frames_missing": int}}
        self.memory = {} 

    def update(self, current_detections):
        active_labels = [d['label'] for d in current_detections]

        # 1. Age the memory of objects we DIDN'T see this frame
        for label in list(self.memory.keys()):
            if label not in active_labels:
                self.memory[label]["frames_missing"] += 1
                
                # If it's been gone too long, forget it completely
                if self.memory[label]["frames_missing"] > self.patience:
                    del self.memory[label]

        # 2. Refresh the memory for objects we DID see
        for det in current_detections:
            self.memory[det['label']] = {
                "box": det['box'], # Save the last known location
                "frames_missing": 0
            }

    def get_buffered_detections(self, current_detections):
        """
        Merges actual frame detections with SENSE's memory.
        """
        buffered = list(current_detections)
        active_labels = [d['label'] for d in current_detections]

        for label, data in self.memory.items():
            # If it's in memory but missing from the current frame, inject it!
            if label not in active_labels and data["frames_missing"] > 0:
                buffered.append({
                    "label": label,
                    "score": 0.0, # Flag that this is a memory, not a live detection
                    "box": data["box"],
                    "is_memory": True 
                })
                
        return buffered