from typing import List
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """
    Represents a single detection instance for a vehicle.
    Includes tracker ID, detection confidence, and bounding box coordinates.
    """
    tracker_id: int = Field(..., description="Unique identifier assigned by the tracker")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score between 0 and 1")
    coords: List[int] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Bounding box coordinates in the format [x1, y1, x2, y2]"
    )


class DetectionData(BaseModel):
    """
    Contains lists of detection items grouped by vehicle type.
    Defaults to empty lists if no detections of that type were found.
    """
    bicycle: List[DetectionItem] = Field(default_factory=list, description="List of detected bicycles")
    car: List[DetectionItem] = Field(default_factory=list, description="List of detected cars")
    bus: List[DetectionItem] = Field(default_factory=list, description="List of detected buses")
    truck: List[DetectionItem] = Field(default_factory=list, description="List of detected trucks")
    motorcycle: List[DetectionItem] = Field(default_factory=list, description="List of detected motorcycles")


class VehicleCounts(BaseModel):
    """
    Contains the confirmed vehicle count for each known vehicle type.
    """
    bicycle: int = Field(..., ge=0, description="Confirmed count of bicycles")
    car: int = Field(..., ge=0, description="Confirmed count of cars")
    bus: int = Field(..., ge=0, description="Confirmed count of buses")
    truck: int = Field(..., ge=0, description="Confirmed count of trucks")
    motorcycle: int = Field(..., ge=0, description="Confirmed count of motorcycles")


class VehicleDetectionResponse(BaseModel):
    """
    Main response model for vehicle detection.
    Includes total count, per-class counts, and detailed detection data.
    """
    total_vehicles: int = Field(..., ge=0, description="Total number of confirmed vehicles detected")
    vehicle_counts: VehicleCounts = Field(..., description="Confirmed vehicle counts by type")
    detection_data: DetectionData = Field(..., description="Detailed detection results grouped by vehicle class")