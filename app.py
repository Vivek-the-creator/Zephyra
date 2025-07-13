import gradio as gr
import cv2
import numpy as np
import uuid
import os
import tempfile
from PIL import Image
from collections import Counter
from models.yolo_model import YOLOv5Detector  # Your detector class

# Create detector instance
detector = YOLOv5Detector(weights='yolov5s.pt', device='cpu')

# Class names for COCO dataset (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def detect_and_caption(image):
    # Convert NumPy BGR image to PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_pil.save(tmp.name)
        tmp_path = tmp.name

    # Detect using image path
    results = detector.detect(tmp_path)

    # Draw boxes on image copy
    img_out = image.copy()
    for det in results:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls_id = det['class']
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_out, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Generate caption
    counts = Counter([COCO_CLASSES[d['class']] for d in results])
    caption = ', '.join(f"{cnt} {cls}" + ("s" if cnt > 1 else "") for cls, cnt in counts.items())
    caption = "Detected objects: " + (caption if caption else "None")

    return img_out, caption

# Gradio interface
iface = gr.Interface(
    fn=detect_and_caption,
    inputs=gr.Image(type='numpy'),
    outputs=[gr.Image(type='numpy'), gr.Textbox()],
    title="ZEPHYRA",
    description="Upload an image to detect objects and get a caption."
)

if __name__ == "__main__":
    iface.launch()
