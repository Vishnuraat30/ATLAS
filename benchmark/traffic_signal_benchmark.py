import random
from copy import deepcopy
from src.DBWSA import TrafficSignalAllocator  
from schema.intersections import IntersectionData
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def generate_random_traffic(mu=30, sigma=10) -> IntersectionData:
    vehicle_types = ["bicycle", "motorcycle", "car", "bus", "truck"]
    intersection_data = {"intersection": {}}

    for i in range(1, 5):  # road1 to road4
        vehicle_counts = {v: max(0, int(random.gauss(mu * (1 if v == "car" else 0.1), sigma))) for v in vehicle_types}
        total_vehicles = sum(vehicle_counts.values())
        road_length = random.choice([200, 250, 275, 300])  # Simulated variability

        intersection_data["intersection"][f"road{i}"] = {
            "vehicle_counts": vehicle_counts,
            "total_vehicles": total_vehicles,
            "road_length": road_length
        }

    IntersectionData.model_validate(intersection_data)
    return intersection_data


def fixed_time_allocation(num_roads=4, cycle_time=60, yellow_time=3) -> Dict[str, Dict[str, int]]:
    green_time = (cycle_time - yellow_time) // num_roads
    signal_times = {}

    for i in range(1, num_roads + 1):
        red_time = cycle_time - (green_time + yellow_time)
        signal_times[f"road{i}"] = {
            "Green": green_time,
            "Yellow": yellow_time,
            "Red": red_time
        }

    return signal_times


def calculate_effective_vehicles(vehicle_counts: Dict[str, int], vehicle_weights: Dict[str, float]) -> float:
    return sum(vehicle_counts.get(v, 0) * vehicle_weights.get(v, 0) for v in vehicle_counts)

def run_benchmark(iterations=500):
    allocator = TrafficSignalAllocator()
    vehicle_weights = allocator.vehicle_weights

    total_dbwsa_weighted_green = 0
    total_fixed_weighted_green = 0

    # Initialize metrics tracking
    metrics = {
        "iteration": [],
        "dbwsa_avg_green": [],
        "fixed_avg_green": [],
        "improvement_percent": []
    }

    # Track max values
    max_dbwsa_avg = -float("inf")
    max_fixed_avg = -float("inf")
    max_improvement = -float("inf")

    max_dbwsa_iter = -1
    max_fixed_iter = -1
    max_improvement_iter = -1

    for i in range(iterations):
        print(f"\nIteration {i + 1}")

        traffic_data = generate_random_traffic()
        traffic_data_copy = deepcopy(traffic_data)

        # DBWSA Allocation
        dbwsa_result = allocator.allocate_signal_times(traffic_data)

        # Fixed Time Allocation
        fixed_result = fixed_time_allocation()

        # Calculate weighted green times
        dbwsa_weighted_sum = 0
        fixed_weighted_sum = 0
        total_effective_vehicles = 0

        for road, data in traffic_data["intersection"].items():
            eff_vehicles = calculate_effective_vehicles(data["vehicle_counts"], vehicle_weights)
            total_effective_vehicles += eff_vehicles

            dbwsa_green = dbwsa_result[road]["Green"]
            fixed_green = fixed_result[road]["Green"]

            dbwsa_weighted_sum += dbwsa_green * eff_vehicles
            fixed_weighted_sum += fixed_green * eff_vehicles

        dbwsa_avg_green = dbwsa_weighted_sum / total_effective_vehicles if total_effective_vehicles else 0
        fixed_avg_green = fixed_weighted_sum / total_effective_vehicles if total_effective_vehicles else 0

        total_dbwsa_weighted_green += dbwsa_avg_green
        total_fixed_weighted_green += fixed_avg_green

        improvement = ((dbwsa_avg_green - fixed_avg_green) / fixed_avg_green * 100) if fixed_avg_green else 0

        # Update max trackers
        if dbwsa_avg_green > max_dbwsa_avg:
            max_dbwsa_avg = dbwsa_avg_green
            max_dbwsa_iter = i + 1

        if fixed_avg_green > max_fixed_avg:
            max_fixed_avg = fixed_avg_green
            max_fixed_iter = i + 1

        if improvement > max_improvement:
            max_improvement = improvement
            max_improvement_iter = i + 1

        # Logging
        print("DBWSA Signal Allocation:")
        for road, timings in dbwsa_result.items():
            print(f"{road}: {timings}")
        print("Fixed-Time Allocation:")
        for road, timings in fixed_result.items():
            print(f"{road}: {timings}")
        print(f"Weighted Avg Green Time per Vehicle:")
        print(f"  DBWSA:  {dbwsa_avg_green:.2f} sec")
        print(f"  Fixed:  {fixed_avg_green:.2f} sec")
        print(f"Improvement of DBWSA over Fixed: {improvement:.2f}%")

        # Store metrics
        metrics["iteration"].append(i + 1)
        metrics["dbwsa_avg_green"].append(dbwsa_avg_green)
        metrics["fixed_avg_green"].append(fixed_avg_green)
        metrics["improvement_percent"].append(improvement)

    # Overall summary
    avg_dbwsa = total_dbwsa_weighted_green / iterations
    avg_fixed = total_fixed_weighted_green / iterations
    overall_improvement = ((avg_dbwsa - avg_fixed) / avg_fixed * 100) if avg_fixed else 0

    print("\n==== Overall Benchmark Summary ====")
    print(f"Average Weighted Green Time (DBWSA): {avg_dbwsa:.2f} sec")
    print(f"Average Weighted Green Time (Fixed) : {avg_fixed:.2f} sec")
    print(f"Overall Improvement: {overall_improvement:.2f}%")

    print("\n==== Maximum Metrics During Simulation ====")
    print(f"Max DBWSA Avg Green:     {max_dbwsa_avg:.2f} sec at Iteration {max_dbwsa_iter}")
    print(f"Max Fixed Avg Green:     {max_fixed_avg:.2f} sec at Iteration {max_fixed_iter}")
    print(f"Max Improvement Percent: {max_improvement:.2f}% at Iteration {max_improvement_iter}")

    # Store overall summary and max metrics
    metrics["avg_dbwsa"] = avg_dbwsa
    metrics["avg_fixed"] = avg_fixed
    metrics["overall_improvement"] = overall_improvement
    metrics["max_dbwsa_avg_green"] = max_dbwsa_avg
    metrics["max_fixed_avg_green"] = max_fixed_avg
    metrics["max_improvement_percent"] = max_improvement
    metrics["max_dbwsa_iteration"] = max_dbwsa_iter
    metrics["max_fixed_iteration"] = max_fixed_iter
    metrics["max_improvement_iteration"] = max_improvement_iter

    return metrics


def plot_cumulative_improvement(metrics: dict):
    iterations = metrics["iteration"]
    improvements = metrics["improvement_percent"]

    # Calculate cumulative average improvement
    cumulative_avg_improvement = np.cumsum(improvements) / np.arange(1, len(improvements) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cumulative_avg_improvement, linestyle='-', color='green', label='Cumulative Avg. Improvement (%)')

    plt.title("Cumulative Improvement of DBWSA Over Fixed-Time Allocation")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Average Improvement (%)")
    plt.grid(True)
    plt.xticks(range(0, len(iterations) + 1, max(1, len(iterations) // 10)))
    plt.tight_layout()
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    metrics = run_benchmark()
    plot_cumulative_improvement(metrics)