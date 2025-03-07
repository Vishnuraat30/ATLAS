# Vehicle Detection and Traffic Analysis using YOLOv8

## Overview

This project uses YOLOv8 for vehicle detection and tracking in traffic videos. The detected vehicles are counted and logged into a JSON file.

## Project Structure

```
project_root/
│── models/
│   └── VehicleDetectionYolov8Model.pt  # Pretrained YOLOv8 model
│── traffic-videos/
│   └── test-video.mp4                   # Place input videos here
│── masks/
│   └── test-video_mask.jpg               # Place mask images here
│── output/
│   └── Results will be stored here
│── script.py                              # Main vehicle detection script
│── generate_mask.py                        # Script to generate mask images
│── requirements.txt                        # Required Python packages
│── README.md                               # This file
```

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python installed (preferably 3.8+). Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Place Input Video

Place your traffic video inside the `traffic-videos/` folder. Example:

```
traffic-videos/test-video.mp4
```

### 3. Generate a Mask Image

A mask is used to filter the region of interest in the video. To generate a mask automatically, run:

```bash
python generate_mask.py --input_video traffic-videos/test-video.mp4 --output_mask masks/test-video_mask.jpg
```

This will create a grayscale mask image where white (255) represents the area to analyze, and black (0) is ignored.

### 4. Run the Vehicle Detector

Execute the script with the following command:

```bash
python script.py --input_video traffic-videos/test-video.mp4 --output_folder output --mask_image masks/test-video_mask.jpg
```

### 5. Output

- **Processed Video**: The annotated video is saved in `output/test-video/`.
- **JSON Report**: Vehicle tracking data is stored in `output/test-video/test-video_traffic_data.json`.

## Example Output

```
Processed video saved in output/test-video/
Traffic data saved to output/test-video/test-video_traffic_data.json
Total confirmed vehicles detected: 50
Car: 30
Bus: 10
Truck: 5
Motorcycle: 5
```

## Notes

- You can adjust detection parameters like confidence threshold using `--confidence_threshold`.
- Change the confirmation frame count using `--confirmation_frame` to tweak detection stability.

## Credits

Developed using Ultralytics YOLOv8 and Supervision Library.

