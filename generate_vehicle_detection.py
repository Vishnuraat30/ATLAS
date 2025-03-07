import cv2
import json
import os
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

# Load YOLO model
model_path = "models/yolo11x.pt"
model = YOLO(model_path) 

# Initialize tracker
tracker = sv.ByteTrack()

# Define vehicle classes
VEHICLE_CLASSES = ['bicycle', 'car', 'bus', 'truck', 'motorbike']

def generate_predictions(input_video_path: str, 
                         output_folder: str, 
                         mask_image_path: str, 
                         confirmation_frame: int = 20,
                         confidence_threshold: float = 0.35) -> json:
    
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    
    output_video_path = os.path.join(video_output_folder, f"{video_name}.avi")
    json_output_path = os.path.join(video_output_folder, f"{video_name}_traffic_data.json")

    cap = cv2.VideoCapture(input_video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Error: Could not read the mask image at {mask_image_path}")

    mask = cv2.resize(mask, (frame_width, frame_height))  
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    vehicle_data = {}
    vehicle_counts = {cls: set() for cls in VEHICLE_CLASSES}  
    confirmed_counts = {cls: set() for cls in VEHICLE_CLASSES}  

    detection_history = defaultdict(lambda: deque(maxlen=confirmation_frame))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
        results = model(masked_frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.confidence > confidence_threshold]
        detections = tracker.update_with_detections(detections)

        labels = [
            f"{model.names[int(cls)]} {conf:.2f} (ID: {tracker_id})"
            for cls, conf, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id)
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        for box, cls, conf, tracker_id in zip(detections.xyxy, detections.class_id, detections.confidence, detections.tracker_id):
            label = model.names[int(cls)]
            if label in VEHICLE_CLASSES and tracker_id is not None:
                x1, y1, x2, y2 = map(int, box)
                
                if label not in vehicle_data:
                    vehicle_data[label] = []

                vehicle_data[label].append({
                    "tracker_id": int(tracker_id),
                    "confidence": float(conf),
                    "coords": [x1, y1, x2, y2]
                })

                detection_history[tracker_id].append(label)

                if len(detection_history[tracker_id]) == confirmation_frame:
                    most_common_label = max(set(detection_history[tracker_id]), key=detection_history[tracker_id].count)
                    if most_common_label == label and tracker_id not in confirmed_counts[label]:
                        confirmed_counts[label].add(tracker_id)

        y_offset = 30  
        text_color = (0, 255, 0)  
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        total_confirmed_vehicles = sum(len(confirmed_counts[cls]) for cls in VEHICLE_CLASSES)

        cv2.putText(frame, f"Total: {total_confirmed_vehicles}", (frame_width - 200, y_offset), font, font_scale, text_color, thickness)

        for cls in VEHICLE_CLASSES:
            y_offset += 30
            count = len(confirmed_counts[cls])
            cv2.putText(frame, f"{cls.capitalize()}: {count}", (frame_width - 200, y_offset), font, font_scale, text_color, thickness)

        out.write(frame)

    cap.release()
    out.release()

    final_output = {
        "total_vehicles": total_confirmed_vehicles,
        "vehicle_counts": {cls: len(confirmed_counts[cls]) for cls in VEHICLE_CLASSES},
        "detection_data": vehicle_data
    }

    with open(json_output_path, "w") as json_file:
        json.dump(final_output, json_file, indent=4)

    print(f"Processed video saved in {video_output_folder}")
    print(f"Traffic data saved to {json_output_path}")
    print(f"Total confirmed vehicles detected: {total_confirmed_vehicles}")
    for cls in VEHICLE_CLASSES:
        print(f"{cls.capitalize()}: {len(confirmed_counts[cls])}")

if __name__ == "__main__":
    input_video_path = "traffic-videos/test-video1.mp4"
    output_folder = "output_yolo12x"
    mask_image_path = "masks/test-video1_mask.jpg"
    CONFIRMATION_FRAMES = 20  
    CONFIDENCE_THRESHOLD = 0.30

    generate_predictions(input_video_path, 
                         output_folder,
                         mask_image_path,
                         CONFIRMATION_FRAMES, 
                         CONFIDENCE_THRESHOLD)