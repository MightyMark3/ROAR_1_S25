import numpy as np
import math


def normalize_rad(rad: float):
    return rad % (2 * np.pi)

class LatController:
    def run(self, vehicle_location, vehicle_rotation, next_waypoint) -> float:
        """
        Calculates the steering command using the pure pursuit algorithm.

        Args:
            vehicle_location (np.array): Current vehicle location [x, y, z].
            vehicle_rotation (float): Current vehicle rotation (yaw) in radians.
            next_waypoint (Waypoint): Next waypoint to track.

        Returns:
            steering_command (float): Steering command in radians.
        """

        # # Calculate vector pointing from vehicle to next waypoint
        # waypoint_vector = np.array(next_waypoint.location) - np.array(vehicle_location)

        # # Project waypoint vector onto heading vector to find lookahead point
        # distance_to_waypoint = abs(np.linalg.norm(waypoint_vector))
        # if distance_to_waypoint == 0:
        #     return 0  # Prevent division by zero

        # waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

        # # Calculate steering command
        # alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(
        #     math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
        # )

        # steering_command = 1.5 * math.atan2(
        #     2.0 * 4.7 * math.sin(alpha) / distance_to_waypoint, 1.0
        # )

        # return float(steering_command)
        
        # Transform the lookahead point to the vehicle's local coordinate frame
        dx = next_waypoint.location[0] - vehicle_location[0]
        dy = next_waypoint.location[1] - vehicle_location[1]

        # Convert global coordinates to local vehicle coordinates
        local_x = dx * math.cos(-vehicle_rotation[2]) - dy * math.sin(-vehicle_rotation[2])
        local_y = dx * math.sin(-vehicle_rotation[2]) + dy * math.cos(-vehicle_rotation[2])

        # Calculate the steering angle using the pure pursuit formula
        curvature = 2 * local_y / (np.linalg.norm(next_waypoint.location - vehicle_location) ** 2)
        steering_angle = -1.5 * np.arctan(4.7 * curvature)
        # print(steering_angle)
        # print(type(steering_angle))
        return steering_angle

