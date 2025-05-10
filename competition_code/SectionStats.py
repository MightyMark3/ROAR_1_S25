from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
import atexit

def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(waypoints)
        d = dist_to_waypoint(location, waypoints[ind])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind

def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)


class SectionStats:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        # vehicle: roar_py_interface.RoarPyActor,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        # self.vehicle = vehicle
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.section_indeces = []
        self.current_waypoint_idx = 0
        self.num_ticks = 0
        self.section_start_ticks = 0
        self.current_section = 0
        self.lapNum = 1
        self.previous_location = None
        self.section_start_distance = 0
        self.current_distance = 0
        self.initialize()
        
    def initialize(self) -> None:
        # # NOTE waypoints are changed through this line
        # self.maneuverable_waypoints = (
        #     roar_py_interface.RoarPyWaypoint.load_waypoint_list(
        #         np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
        #     )[35:]
        # )

        sectionLocations = [
            [-278, 372], # Section 0 start location
            [64, 890], # Section 1 start location
            [511, 1037], # Section 2 start location
            [762, 908], # Section 3 start location
            [198, 307], # Section 4 start location
            [-11, 60], # Section 5 start location
            [-85, -339], # Section 6 start location
            [-210, -1060], # Section 7 start location 
            [-318, -991], # Section 8 start location
            [-352, -119], # Section 9 start location
        ]
        # for i in sectionLocations:
        #     self.section_indeces.append(
        #         findClosestIndex(i, self.maneuverable_waypoints)
        #     )
        self.section_indeces = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]

        # print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        # print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        # print(f"Section indexes: {self.section_indeces}")
        # print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

    def step(self) -> None:
        """
        This function is called every world step.
        """
        self.num_ticks += 1

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        # vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        # vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        # current_speed_kmh = vehicle_velocity_norm * 3.6
        if self.previous_location is not None:
            self.current_distance += np.linalg.norm(vehicle_location - self.previous_location)
        self.previous_location = vehicle_location

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        # print(f"STA loc {vehicle_location} ind {self.current_waypoint_idx}")

        # compute and print section timing
        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                print(f"Section {i}: ticks {(self.num_ticks - self.section_start_ticks):4d}  distance {(self.current_distance - self.section_start_distance):6.1f}")
                self.section_start_ticks = self.num_ticks
                self.section_start_distance = self.current_distance
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1
                    print(f"\nLap {self.lapNum}\n")
