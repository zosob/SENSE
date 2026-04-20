class LogicGate:
    def __init__(self, threshold=0.70):
        self.threshold = threshold

    def audit(self, claims, detections):
        report = []
        # Create a dictionary of verified objects for easy lookup
        # This filters out low-confidence detections immediately
        verified_objs = {d['label']: d for d in detections if d['score'] >= self.threshold}

        for claim in claims:
            # RELATIONSHIP CHECK: e.g., "cup on desk"
            if " on " in claim:
                target, surface = claim.split(" on ")
                
                if target in verified_objs and surface in verified_objs:
                    # Trigger your check_support method!
                    supported = self.check_support(verified_objs[target]['box'], verified_objs[surface]['box'])
                    
                    status = "VERIFIED" if supported else "SPATIAL_HALLUCINATION"
                    report.append({"claim": claim, "status": status})
                else:
                    report.append({"claim": claim, "status": "HALLUCINATION", "reason": "One or both objects missing"})

            # EXISTENCE CHECK: e.g., "cup"
            else:
                if claim in verified_objs:
                    report.append({"claim": claim, "status": "VERIFIED", "score": verified_objs[claim]['score']})
                else:
                    report.append({"claim": claim, "status": "HALLUCINATION", "reason": "Object not found above threshold"})
                    
        return report
    
    def check_support(self, object_box, surface_box):
        """
        Returns True if the object is physically 'on top of' the surface.
        """
        obj_ymin, obj_xmin, obj_ymax, obj_xmax = object_box.values()
        sur_ymin, sur_xmin, sur_ymax, sur_xmax = surface_box.values()
        
        # Simple Logic: Is the bottom of the object near the top of the surface?
        if obj_ymax >= sur_ymin and obj_xmin >= sur_xmin and obj_xmax <= sur_xmax:
            return True
        return False