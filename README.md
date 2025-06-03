# 🛰️ ATLAS – Adaptive Traffic Light Allocation System

## Overview

**ATLAS** is an intelligent traffic analysis and signal control system that performs real-time vehicle detection, tracking, and classification using **YOLOv8** and **ByteTrack**. It analyzes traffic videos to count vehicles within defined regions, generating actionable insights for urban mobility.

Powered by the **Density-Based Weighted Allocation (DBWA)** algorithm, ATLAS dynamically adjusts traffic light durations (green, yellow, red) based on real-time vehicle density and type—ensuring smoother flow, reduced congestion, and smarter signal control.


### **Key Features**  
✔ Detects and tracks multiple vehicle types including cars, buses, trucks, motorcycles, and bicycles.  
✔ Uses a mask to focus detection on areas of interest for precise analysis.  
✔ Confirms vehicle presence over multiple frames to improve classification accuracy.  
✔ Outputs annotated video with bounding boxes, labels, and live vehicle counts.  
✔ Generates detailed JSON reports summarizing vehicle counts and detection data.  
✔ Fully configurable via YAML for flexible model, input, and output management.

Ideal for smart traffic monitoring systems and adaptive signal control implementations.


## Project Structure

```
project_root/
│── core/
│   └── config.py                           # Pretrained YOLOv8 model
│── masks/
│   └── test-video_mask.jpg                 # Place mask images here
│── models/
│   └── VehicleDetectionYolov8Model.pt      # Pretrained YOLOv8 model
│   └── VehicleDetectionYolov11LModel.pt    # Pretrained YOLOv8 model
│── output/
│   └── Results will be stored here
│── schema/  
│   └── detections.py                       # Schema for Vehicle Detection
│   └── intersections.py                    # Schema for Traffic Signal Intersection Data
│── src/  
│   └── DBWSA.py                            # Traffic Signal Optimization Algorithm
│   └── generate_mask.py                    # Script to generate mask images
│   └── run_detector.py                     # CMD Script for Vehicle Detection
│   └── get_predictions.py                  # Script to generate detection(without mask)
│── traffic_data/
│   └── Intersection Data
│── traffic-videos/
│   └── test-video.mp4                      # Place input videos here
│── config.yaml                             # Config file              
│── requirements.txt                        # Required Python packages
│── README.md                               
│── preview.gif                            
``` 

<p align="center">
    <img src="preview.gif">
</p>


## Setup Instructions

### 1. Clone the Repository

Ensure you have Python(preferably 3.8+) & git installed. Then, install the required dependencies:

First, clone the repository using Git:

```bash
git https://github.com/saketjha34/ATLAS.git
cd ATLAS
```

### 2. Install Dependencies

Ensure you have Python installed (preferably 3.8+). Then, install the required dependencies:

```bash
python -m venv venv
```

```bash
venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

### 3. Congigure config.yaml

This project is configured via a `config.yaml` file. Below is a breakdown of the key configuration options and how to use them.

### 📁 `config.yaml` Structure

```yaml
detection:
  input_traffic_video_path: "traffic-videos/test-video.mp4"       # Path to the input traffic video
  output_folder_name: "output"                                     # Directory to save output video and data
  mask_image_path: "masks/test-video_mask.jpg"                     # Grayscale mask image to isolate the road area
  confirmation_frame: 15                                           # Number of consistent frames to confirm a vehicle
  confidence_threshold: 0.35                                       # Minimum confidence score for detections
  yolo_model:
    yolo_model_path: "models/VehicleDetectionYolov11LModel.pt"     # Path to the YOLOv8 model file

traffic_signal_allocator:
  traffic_intersection_data_path: "traffic_data/intersection_data.json"  # Path to the JSON file containing traffic data
  base_green_time: 5                                                    # Minimum green signal time (in seconds)
  max_green_time: 60                                                    # Maximum green signal time (in seconds)
  yellow_time: 3                                                        # Fixed yellow light duration (in seconds)
  total_cycle_time: 60                                                  # Total cycle time for one signal phase (in seconds)
```

### ✅ Usage

1. **Update paths** as per your directory structure for video, model, mask, and traffic data.
2. Ensure the model and mask exist at specified paths before running.
3. Run the main script:

This will:

* Detect and track vehicles from the video.
* Output an annotated video and JSON summary.
* Use the DBWA algorithm to simulate adaptive signal allocation.


### 4. Place Input Video

Place your traffic video inside the `traffic-videos/` folder.


### 5. Generate a Mask Image

A mask is used to filter the region of interest in the video. To generate a mask automatically, run:

```bash
python -m src.generate_mask
```

This will create a grayscale mask image where white (255) represents the area to analyze, and black (0) is ignored.

### 5. Run the Vehicle Detector

Execute the script with the following command:

```bash
python -m src.run_detector
```

### 6. Run the Traffic Signal Optimization Algorithm

#### **Parameters & Customization**  Congiure it in `config.yaml` file

The algorithm uses the following default parameters, which can be modified:  

| Parameter          | Description                         | Default |
|------------------  |-------------------------------------|---------|
| `BASE_GREEN_TIME`  | Minimum green signal duration       | 5s      |
| `MAX_GREEN_TIME`   | Maximum green signal duration       | 60s     |
| `YELLOW_TIME`      | Fixed yellow signal duration        | 3s      |
| `TOTAL_CYCLE_TIME` | Total signal cycle duration         | 60s     |

You can modify these values in `DBWSA.py`.  

```bash
python src.DBWSA.py 
```

### 7. Output

- **Processed Video**: The annotated video is saved in `output/test-video/`.
- **JSON Report**: Vehicle tracking data is stored in `output/test-video/test-video_traffic_data.json`.

## Example Output

```
Validation successful
Processed video saved in output\test-video
Traffic data saved to output\test-video\test-video_traffic_data.json
Total confirmed vehicles detected: 14
Bicycle: 0
Car: 12
Bus: 0
Truck: 2
Motorcycle: 0

Optimized Traffic Signal Timings:
🚦 Traffic Signal Allocation Results:

➡ ROAD1:
   🟢 Green  : 28 seconds
   🟡 Yellow :  3 seconds
   🔴 Red    : 29 seconds
   ⏱️ Total  : 60 seconds

➡ ROAD2:
   🟢 Green  :  9 seconds
   🟡 Yellow :  3 seconds
   🔴 Red    : 48 seconds
   ⏱️ Total  : 60 seconds

➡ ROAD3:
   🟢 Green  : 12 seconds
   🟡 Yellow :  3 seconds
   🔴 Red    : 45 seconds
   ⏱️ Total  : 60 seconds

➡ ROAD4:
   🟢 Green  : 11 seconds
   🟡 Yellow :  3 seconds
   🔴 Red    : 46 seconds
   ⏱️ Total  : 60 seconds
```

## Notes

- You can adjust detection parameters like confidence threshold using `--confidence_threshold`.
- Change the confirmation frame count using `--confirmation_frame` to tweak detection stability.

## Credits

Developed using Ultralytics and Supervision Library.