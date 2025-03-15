# Vehicle Detection and Traffic Analysis using YOLOv8

## Overview

This project uses YOLOv8 for vehicle detection and tracking in traffic videos. The detected vehicles are counted and logged into a JSON file.

## Project Structure

```
project_root/
│── models/
│   └── VehicleDetectionYolov8Model.pt  # Pretrained YOLOv8 model
│   └── VehicleDetectionYolov11LModel.pt  # Pretrained YOLOv8 model
│── traffic-videos/
│   └── test-video.mp4                   # Place input videos here
│── masks/
│   └── test-video_mask.jpg               # Place mask images here
│── output/
│   └── Results will be stored here
│── generate_vehicle_detection.py           # Main vehicle detection script
│── run_detector.py                         # CMD Script for Vehicle Detection
│── generate_mask.py                        # Script to generate mask images
│── get_predictions.py                      # Script to generate detection(without mask)
│── requirements.txt                        # Required Python packages
│── README.md                               # This file
```

<p align="center">
    <img src="../masks/test-video4_mask.jpg">
</p>

## Setup Instructions

### 1. Clone the Repository

First, clone the repository using Git:

```bash
git clone https://github.com/saketjha34/Traffic-Management-System.git
cd Traffic-Management-System
```

### 2. Install Dependencies

Ensure you have Python installed (preferably 3.8+). Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Place Input Video

Place your traffic video inside the `traffic-videos/` folder. Example:

```
traffic-videos/test-video.mp4
```

### 4. Generate a Mask Image

A mask is used to filter the region of interest in the video. To generate a mask automatically, run:

```bash
python generate_mask.py 
```

This will create a grayscale mask image where white (255) represents the area to analyze, and black (0) is ignored.

### 5. Run the Vehicle Detector

Execute the script with the following command:

```bash
python run_detector.py --input_video traffic-videos/test-video1.mp4 --output_folder output --mask_image masks/test-video1_mask.jpg --confirmation_frame 15 --confidence_threshold 0.35
```

### 6. Output

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

Developed using Ultralytics and Supervision Library.