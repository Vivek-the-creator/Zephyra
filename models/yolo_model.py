import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# Add yolov5 and yolov5/models to path
FILE = Path(__file__).resolve()
YOLOV5_DIR = FILE.parents[1] / 'yolov5'
sys.path.insert(0, str(YOLOV5_DIR))
sys.path.insert(0, str(YOLOV5_DIR / 'models'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device



class YOLOv5Detector:
    def __init__(self, weights='yolov5s.pt', device='cpu'):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.model.eval()

    def detect(self, image_path):
        # Load image
        img0 = cv2.imread(image_path)
        assert img0 is not None, f'Image Not Found {image_path}'

        # Preprocess
        img = letterbox(img0, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # Normalize
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        # Process detections
        results = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    results.append({
                        'bbox': [int(x.item()) for x in xyxy],
                        'confidence': round(conf.item(), 3),
                        'class': int(cls.item())
                    })
        return results
