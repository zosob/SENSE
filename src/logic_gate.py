class LogicGate:
    def __init__(self, threshold=0.70):
        self.threshold = threshold

    def audit(self, claims, detections):
        """
        Compares LLM claims against Vision Probe detections.
        """
        report = []
        for claim in claims:
            # Find the best detection match for this specific claim
            matches = [d for d in detections if d['label'] == claim]
            
            if not matches:
                report.append({"claim": claim, "status": "HALLUCINATION", "reason": "Object not found"})
                continue
            
            best_match = max(matches, key=lambda x: x['score'])
            
            if best_match['score'] >= self.threshold:
                report.append({"claim": claim, "status": "VERIFIED", "score": best_match['score']})
            else:
                report.append({"claim": claim, "status": "UNCERTAIN", "score": best_match['score']})
                
        return report