import argparse
import os
from typing import Sequence
import onnxruntime as ort
import cv2
import numpy as np
import structlog

from tqdm import tqdm

_logger = structlog.get_logger()

YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class BlurStuff:
    def __init__(self, model_path: str, classes: Sequence[str] = ["person"], blur_ratio: float = 50, nms_threshold: float = 0.25):
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 2  # 0 for verbose
        sess_options.enable_profiling = False
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        
        _logger.info(f"Running with providers: {providers}")
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.blur_ratio = blur_ratio
        self.nms_threshold = nms_threshold
        
        for c in classes:
            if c not in YOLO_CLASSES:
                raise ValueError(f"Invalid class: {c}")
        self.classes = classes

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = np.argsort(scores)[::-1]
        kept_indices = []
        
        while len(indices) > 0:
            current_idx = indices[0]
            kept_indices.append(current_idx)
            remaining_indices = indices[1:]
            ious = [self.calculate_iou(boxes[current_idx], boxes[idx]) for idx in remaining_indices]
            indices = [idx for i, idx in enumerate(remaining_indices) 
                    if ious[i] < iou_threshold]
        
        return boxes[kept_indices], scores[kept_indices]
    
    def parse_row(self, row: np.ndarray, height: int, width: int) -> np.ndarray:
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * width
        y1 = (yc - h/2) / 640 * height
        x2 = (xc + w/2) / 640 * width
        y2 = (yc + h/2) / 640 * height
        prob = row[4:].max()
        class_id = row[4:].argmax()
        label = YOLO_CLASSES[class_id]  
        
        return [x1, y1, x2, y2, label, prob]
    
    def blur(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[0], image.shape[1]
    
        img = cv2.resize(image, self.input_shape[2:])
        img = img.transpose(2, 0, 1)
        img = img.reshape(self.input_shape)
        img = img / 255.0
        img = img.astype("float32")
        
        outputs = self.session.run([self.output_name], { self.input_name: img })

        output = outputs[0].squeeze().transpose()
        
        boxes = [row for row in [self.parse_row(row, height, width) for row in output] if row[5] > self.nms_threshold and row[4] == "person"]  
        boxes, _ = self.non_max_suppression([box[:4] for box in boxes], [box[5] for box in boxes])
        
        for box in boxes:
            obj = image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            blur_obj = cv2.blur(obj, (self.blur_ratio, self.blur_ratio))
            image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            
        return image
    
def process_image(image, blur_stuff, output_path, show):
    blurred = blur_stuff.blur(image)
    cv2.imwrite(output_path, blurred)
    if show:
        cv2.imshow("blurred stuff", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(video_path, blur_stuff, output_path, show):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _logger.error(f"Error opening video file: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    with tqdm(total=total_frames, desc=f"Processing video {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            blurred_frame = blur_stuff.blur(frame)
            out.write(blurred_frame)
            if show:
                cv2.imshow("blurred stuff", blurred_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            pbar.update(1)
    
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
        
def get_output_path(input_path: str, output_dir: str) -> str:
    if output_dir:
        return os.path.join(output_dir, os.path.basename(input_path))
    else:
        return os.path.join(os.path.dirname(input_path), f"blurred_{os.path.basename(input_path)}")
    
def add_file_to_set(file: str, input_path: str, images: set[str], videos: set[str]):
    full_path = os.path.join(input_path, file)
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        images.add(full_path)
    elif file.lower().endswith(('.mp4', '.avi', '.mov')):
        videos.add(full_path)

def main():    
    parser = argparse.ArgumentParser(description="Blur stuff in images and videos")
    parser.add_argument("-i", "--input", required=True, nargs='+', help='Input images, videos, or directories')
    parser.add_argument("-o", "--output_dir", help='Output directory for processed files')
    parser.add_argument("-m", "--model", default='models/yolov8n.onnx', help='Path to ONNX model file')
    parser.add_argument("-s", "--show", action='store_true', help='Show results')
    args = parser.parse_args()

    blur_stuff = BlurStuff(args.model)
    
    images = set()
    videos = set()
    for input_path in args.input:
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                add_file_to_set(file, input_path, images, videos)
        elif os.path.isfile(input_path):
            add_file_to_set(os.path.basename(input_path), os.path.dirname(input_path), images, videos)
        else:
            _logger.error(f"Invalid input: {input_path}")
            
    _logger.debug(f"images: {images}, videos: {videos}")
    
    output_dir = args.output_dir if args.output_dir else None
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tqdm(total=len(images), desc="Processing images") as pbar:
        for image in images:
            img = cv2.imread(image)
            if img is None:
                _logger.error(f"Error reading image: {image}")
                continue
            output_path = get_output_path(image, output_dir)
            process_image(img, blur_stuff, output_path, args.show)
            pbar.update(1)
    
    for video in videos:
        output_path = get_output_path(video, output_dir)
        process_video(video, blur_stuff, output_path, args.show)

if __name__ == "__main__":
    main()