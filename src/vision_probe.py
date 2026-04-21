import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class VisionProbe:
    def __init__(self):
        # Using a more robust model checkpoint
        self.model_id = "IDEA-Research/grounding-dino-tiny" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def probe_batch(self, image_paths, text_queries):
        """
        Processes multiple images at once to saturate the Blackwell GPU.
        """
        from PIL import Image
        images = [Image.open(path).convert("RGB") for path in image_paths]
        
        # Grounding DINO handles text as a single string separated by dots
        query = ". ".join(text_queries) + "."
        
        inputs = self.processor(images=images, text=[query]*len(images), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Inside your probe_batch function after getting 'outputs'
        results = []
        target_sizes = torch.tensor([img.size[::-1] for img in images])
        processed_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25, # Neural sensitivity
            text_threshold=0.25,
            target_sizes=target_sizes
        )[0] # Taking the first image result for now

        for score, label, box in zip(processed_results["scores"], processed_results["labels"], processed_results["boxes"]):
            results.append({
                "label": label,
                "score": score.item(),
                "box": {
                    "ymin": box[1].item(),
                    "xmin": box[0].item(),
                    "ymax": box[3].item(),
                    "xmax": box[2].item()
                }
            })

        return results # Now this is a list of dictionaries!