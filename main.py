from models.yolo_model import YOLOv5Detector
from utils.scene_graph import get_scene_relations
from models.gpt_captioner import CaptionGenerator

if __name__ == "__main__":
    image_path = "data/sample_image.jpg"

    detector = YOLOv5Detector()
    detections = detector.detect_objects(image_path)

    print("[INFO] Detected objects:\n", detections[['name', 'confidence']])

    relations = get_scene_relations(detections)
    print("[INFO] Spatial Relations:\n", relations)

    captioner = CaptionGenerator()
    caption = captioner.generate_caption(relations)
    print("[INFO] Scene Description:\n", caption)
