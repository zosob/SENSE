import cv2

class Visualizer:
    def __init__(self, output_dir="data/"):
        self.output_dir = output_dir

    def draw_results(self, image_path, detections, report, output_name="sense_output.jpg"):
        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return
            
        h, w, _ = img.shape

        # We match detections to report items by label
        for item in report:
            label = item['claim']
            # Find the detection that matches this claim
            # (In a relationship claim like 'plant on desk', we look for the primary object)
            target_label = label.split(" on ")[0] if " on " in label else label
            
            match = next((d for d in detections if d['label'] == target_label), None)
            
            if match:
                box = match['box']
                # Convert normalized coordinates to pixel integers
                # Note: Grounding DINO usually returns [xmin, ymin, xmax, ymax]
                # Our unpacker saved them as ymin, xmin, ymax, xmax
                start_point = (int(box['xmin']), int(box['ymin']))
                end_point = (int(box['xmax']), int(box['ymax']))

                # BGR Colors: Green for Verified, Red for Hallucination, Yellow for Uncertain
                color = (0, 255, 0) if item['status'] == "VERIFIED" else (0, 0, 255)
                if item['status'] == "UNCERTAIN": color = (0, 255, 255)

                # Draw the rectangle
                cv2.rectangle(img, start_point, end_point, color, 3)
                
                # Draw the label background and text
                text = f"{label}: {item['status']}"
                cv2.putText(img, text, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save the result
        output_path = f"{self.output_dir}{output_name}"
        cv2.imwrite(output_path, img)
        print(f"\n[Visualizer] Audit image saved to: {output_path}")