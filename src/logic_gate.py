class LogicGate:
    def __init__(self, threshold=0.70):
        self.threshold = threshold

    # Add this to your LogicGate class
    CONSTRAINTS = {
        "mouse": {"requires": "desk", "relation": "on"},
        "laptop": {"requires": "desk", "relation": "on"},
        "plant": {"requires": ["desk", "shelf", "table"], "relation": "on"}
    }
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
        self.verify_dependencies(report, verified_objs)            
        return report
    
    def check_support(self, obj_box, surface_box):
        # Add a small vertical buffer (0.1) so the plant doesn't 
        # have to perfectly touch the desk pixel-to-pixel.
        buffer = (surface_box['ymax'] - surface_box['ymin']) * 0.1
        
        is_above = obj_box['ymax'] >= (surface_box['ymin'] - buffer)
        is_within_width = (obj_box['xmin'] >= surface_box['xmin'] and 
                        obj_box['xmax'] <= surface_box['xmax'])
        
        return is_above and is_within_width
    
    def verify_dependencies(self, report, verified_objs):
        for item in report:
            claim = item['claim']
            if claim in self.CONSTRAINTS:
                requirement = self.CONSTRAINTS[claim]['requires']
                
                # Check if the requirement (e.g., 'desk') is in our verified list
                if isinstance(requirement, list):
                    has_requirement = any(r in verified_objs for r in requirement)
                else:
                    has_requirement = requirement in verified_objs
                    
                if not has_requirement:
                    item['status'] = "LOGICAL_ERROR"
                    item['reason'] = f"Missing dependency: {requirement}"
                    
        return report