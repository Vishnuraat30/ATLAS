"""
Traffic Signal Optimization using Density-Based Weighted Signal Allocation (DBWSA) Algorithm

This module implements an adaptive traffic signal timing system based on vehicle density and type. 
It dynamically adjusts green, yellow, and red signal durations based on real-time traffic data.

Algorithm:
-----------
1. **Traffic Density Calculation**:  
   - Computes vehicle density for each road as:
     ```
     density = total_vehicles / road_length
     ```
   - This ensures longer roads with fewer vehicles do not get unnecessarily long green signals.

2. **Effective Vehicle Count Calculation**:  
   - Assigns weightage to different vehicle types:
     ```
     bicycle: 0.5, motorcycle: 0.5, car: 1, bus: 3, truck: 3
     ```
   - Computes weighted vehicle count for each road.

3. **Green Time Allocation**:  
   - Proportional green signal time is assigned based on road traffic density:
     ```
     green_time = max(BASE_GREEN_TIME, (density / total_density) * MAX_GREEN_TIME)
     ```
   - Ensures minimum and maximum bounds are respected.

4. **Red and Yellow Time Calculation**:  
   - Yellow time is fixed as input by the user.
   - Red time is computed based on total cycle duration:
     ```
     red_time = TOTAL_CYCLE_TIME - (green_time + YELLOW_TIME)
     ```

Parameters:
-----------
- `traffic_data (dict)`: JSON data containing:
    - `total_vehicles (int)`: Total number of vehicles on each road.
    - `vehicle_counts (dict)`: Breakdown of vehicle types and their counts.
    - `road_length (float)`: Length of the road in meters.

- `BASE_GREEN_TIME (int)`: Minimum duration of green light (default: 5s).
- `MAX_GREEN_TIME (int)`: Maximum duration of green light (default: 60s).
- `YELLOW_TIME (int)`: Duration of yellow light (default: 3s).
- `TOTAL_CYCLE_TIME (int)`: Total signal cycle time for an intersection (default: 60s).

Returns:
-----------
A dictionary containing:
- `"Green"`: Allocated green time for each road.
- `"Yellow"`: Fixed yellow signal time.
- `"Red"`: Calculated red signal time.
"""



import json

class TrafficSignalAllocator:
    def __init__(self, base_green_time=5, max_green_time=60, yellow_time=3, total_cycle_time=60):
        """
        Initializes the traffic signal allocator with default or user-defined timing parameters.
        """
        self.base_green_time = base_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.total_cycle_time = total_cycle_time

        # Define weightage for different vehicle types
        self.vehicle_weights = {
            "bicycle": 0.5,
            "motorcycle": 0.5,
            "car": 1,
            "bus": 3,
            "truck": 3
        }

    def calculate_effective_vehicles(self, vehicle_counts):
        """Calculate the effective number of vehicles using predefined weights."""
        return sum(vehicle_counts.get(vehicle, 0) * weight for vehicle, weight in self.vehicle_weights.items())

    def calculate_traffic_density(self, total_vehicles, road_length):
        """Calculate traffic density as vehicles per meter."""
        return total_vehicles / road_length if road_length > 0 else 0

    def allocate_signal_times(self, traffic_data):
        """
        Allocates green, yellow, and red light durations for each road based on traffic density.
        """
        road_densities = {}

        # Validate input format
        if "intersection" not in traffic_data:
            raise ValueError("Invalid JSON format: 'intersection' key missing.")

        # Calculate traffic density for each road
        for road, data in traffic_data["intersection"].items():
            total_vehicles = data["total_vehicles"]
            road_length = data["road_length"]
            road_densities[road] = self.calculate_traffic_density(total_vehicles, road_length)

        # Calculate total density
        total_density = sum(road_densities.values())

        # Allocate green times based on traffic density
        green_times = {}
        for road, density in road_densities.items():
            if total_density == 0:
                green_times[road] = self.base_green_time  # Assign minimum time if no vehicles
            else:
                green_time = max(self.base_green_time, (density / total_density) * self.max_green_time)
                green_time = min(green_time, self.max_green_time)  # Ensure it doesn't exceed max
                green_times[road] = round(green_time)  # Round to nearest second

        # Calculate red times
        signal_times = {}
        for road, green_time in green_times.items():
            red_time = self.total_cycle_time - (green_time + self.yellow_time)
            signal_times[road] = {
                "Green": green_time,
                "Yellow": self.yellow_time,
                "Red": red_time
            }

        return signal_times

# Example usage
if __name__ == "__main__":
    # Load JSON data from file
    with open("traffic_data/intersection_data.json", "r") as file:
        traffic_data = json.load(file)

    # Initialize the TrafficSignalAllocator with default values
    allocator = TrafficSignalAllocator()
    signal_times = allocator.allocate_signal_times(traffic_data)

    # Print results
    print(json.dumps(signal_times, indent=4))