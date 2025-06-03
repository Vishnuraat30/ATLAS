from pydantic import BaseModel, Field
from typing import Dict
from schema.detections import VehicleCounts


class RoadData(BaseModel):
    """
    Contains traffic data for a single road at an intersection.
    """
    total_vehicles: int = Field(..., ge=0, description="Total number of confirmed vehicles on this road")
    road_length: int = Field(..., gt=0, description="Length of the road in meters")
    vehicle_counts: VehicleCounts = Field(..., description="Detailed vehicle counts by type")


class IntersectionData(BaseModel):
    """
    Main response model representing traffic data across multiple roads in an intersection.
    """
    intersection: Dict[str, RoadData] = Field(
        ..., 
        description="Dictionary of road names to their corresponding traffic data"
    )