from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class YoloModelConfig(BaseSettings):
    yolo_model_path: Path = Field(..., description="Path to pretrained YOLO model")


class DetectionConfig(BaseSettings):
    input_traffic_video_path: Path = Field(..., description="Path to input traffic video file")
    output_folder_name: Path = Field(..., description="Folder to save output")
    mask_image_path: Path = Field(..., description="Path to mask image file")
    confirmation_frame: int = Field(20, description="Frames to confirm detection")
    confidence_threshold: float = Field(0.35, description="Confidence threshold")
    yolo_model: YoloModelConfig


class TrafficSignalAllocatorConfig(BaseSettings):
    traffic_intersection_data_path: Path = Field(..., description="Traffic Intersection Data Path")
    base_green_time: int = Field(..., description="Base green light time in seconds")
    max_green_time: int = Field(..., description="Max green light time in seconds")
    yellow_time: int = Field(..., description="Yellow light time in seconds")
    total_cycle_time: int = Field(..., description="Total traffic light cycle time in seconds")