"""
Traffic Signal Optimization using Density-Based Weighted Signal Allocation (DBWSA) Algorithm

This module implements an adaptive traffic signal timing system using real-time traffic data.
It dynamically adjusts green, yellow, and red signal durations based on traffic density and vehicle type.

Traffic intersection data must conform to the schema defined in:
    from schema.intersections import IntersectionData
"""

import json
import yaml 
from typing import Dict, Any
from core.config import TrafficSignalAllocatorConfig
from schema.intersections import IntersectionData


class TrafficSignalAllocator:
    def __init__(self, base_green_time: int = 5, max_green_time: int = 60, yellow_time: int = 3, total_cycle_time: int = 60):
        """
        Initializes the traffic signal allocator with timing configuration.
        
        Args:
            base_green_time (int): Minimum green signal duration in seconds.
            max_green_time (int): Maximum green signal duration in seconds.
            yellow_time (int): Fixed yellow signal duration in seconds.
            total_cycle_time (int): Total duration of one complete signal cycle.
        """
        self.base_green_time = base_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.total_cycle_time = total_cycle_time

        # Weightage for different vehicle types
        self.vehicle_weights = {
            "bicycle": 0.5,
            "motorcycle": 0.5,
            "car": 1,
            "bus": 3,
            "truck": 3
        }

    def calculate_effective_vehicles(self, vehicle_counts: Dict[str, int]) -> float:
        """
        Calculate the weighted number of vehicles on a road segment.
        
        Args:
            vehicle_counts (dict): A dictionary of vehicle types and their counts.
        
        Returns:
            float: Effective vehicle count.
        """
        return sum(vehicle_counts.get(vehicle, 0) * weight for vehicle, weight in self.vehicle_weights.items())

    def calculate_traffic_density(self, total_vehicles: int, road_length: float) -> float:
        """
        Compute vehicle density as vehicles per meter.

        Args:
            total_vehicles (int): Total number of vehicles.
            road_length (float): Length of the road in meters.

        Returns:
            float: Traffic density.
        """
        return total_vehicles / road_length if road_length > 0 else 0

    def allocate_signal_times(self, traffic_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """
        Allocates green, yellow, and red light durations per road based on traffic densities.
        Input must conform to the schema defined in `schema.intersections.IntersectionData`.

        Args:
            traffic_data (dict): Validated traffic data containing 'intersection' dictionary.
        
        Returns:
            dict: Mapping of road names to their respective green, yellow, and red durations.
        
        Raises:
            ValueError: If 'intersection' key is missing from traffic data.
        """
        if "intersection" not in traffic_data:
            raise ValueError("Invalid JSON format: 'intersection' key missing. Ensure input matches IntersectionData schema.")

        road_densities = {}
        for road, data in traffic_data["intersection"].items():
            total_vehicles = data["total_vehicles"]
            road_length = data["road_length"]
            road_densities[road] = self.calculate_traffic_density(total_vehicles, road_length)

        total_density = sum(road_densities.values())

        green_times = {}
        for road, density in road_densities.items():
            if total_density == 0:
                green_times[road] = self.base_green_time
            else:
                proportional_time = (density / total_density) * self.max_green_time
                green_times[road] = round(min(max(self.base_green_time, proportional_time), self.max_green_time))

        signal_times = {}
        for road, green_time in green_times.items():
            red_time = self.total_cycle_time - (green_time + self.yellow_time)
            signal_times[road] = {
                "Green": green_time,
                "Yellow": self.yellow_time,
                "Red": red_time
            }

        return signal_times


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)

    traffic_signal_allocator_config = TrafficSignalAllocatorConfig(**data["traffic_signal_allocator"])

    # Initialize allocator with parameters
    allocator = TrafficSignalAllocator(
        base_green_time=traffic_signal_allocator_config.base_green_time,
        max_green_time=traffic_signal_allocator_config.max_green_time,
        yellow_time=traffic_signal_allocator_config.yellow_time,
        total_cycle_time=traffic_signal_allocator_config.total_cycle_time
    )

    # Load and validate traffic intersection data
    with open(str(traffic_signal_allocator_config.traffic_intersection_data_path), "r") as file:
        traffic_data = json.load(file)

    try:
        # Must conform to the schema defined in schema/intersections.py
        IntersectionData.model_validate(traffic_data)
        print("Intersection Data Validation Successful.")
    except Exception as e:
        raise ValueError("Intersection Data Validation Failed: Match the schema in schema/intersections.py") from e

    # Allocate signal times
    signal_times = allocator.allocate_signal_times(traffic_data)

    # Print formatted results
    print("\n🚦 Traffic Signal Allocation Results:\n")
    for road, timings in signal_times.items():
        green = timings["Green"]
        yellow = timings["Yellow"]
        red = timings["Red"]

        print(f"➡ {road.upper()}:")
        print(f"   🟢 Green  : {green:2d} seconds")
        print(f"   🟡 Yellow : {yellow:2d} seconds")
        print(f"   🔴 Red    : {red:2d} seconds")
        print(f"   ⏱️ Total  : {green + yellow + red:2d} seconds\n")