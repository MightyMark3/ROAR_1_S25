import numpy as np
import math
from collections import deque
from SpeedData import SpeedData
import roar_py_interface


def distance_p_to_p(
    p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint
):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])


class ThrottleController:
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self):
        self.max_radius = 10000
        self.max_speed = 300
        self.intended_target_distance = [0, 30, 60, 90, 120, 140, 170]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0
        self.currentCorner = 0

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")

    def run(
        self, waypoints, current_location, current_speed, current_section, corners
    ) -> (float, float, int):
        self.tick_counter += 1
        distanceToCorner = np.linalg.norm(
            current_location - corners[self.currentCorner]["startLoc"]
        )

        if distanceToCorner < 5:
            self.currentCorner = (self.currentCorner + 1) % len(corners)
            distanceToCorner = np.linalg.norm(
                corners[self.currentCorner]["startLoc"] - current_location
            )

        throttle, brake = self.get_throttle_and_brake(
            current_location,
            current_speed,
            current_section,
            waypoints,
            corners[self.currentCorner],
            distanceToCorner,
        )

        # gear = max(1, (int)(math.log(current_speed + 0.00001, 5)))
        gear = max(1, int(current_speed / 60))
        if throttle < 0:
            gear = -1

        # self.dprint("--- " + str(throttle) + " " + str(brake)
        #             + " steer " + str(steering)
        #             + "     loc x,z" + str(self.agent.vehicle.transform.location.x)
        #             + " " + str(self.agent.vehicle.transform.location.z))

        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        # throttle = 0.05 * (100 - current_speed)
        return throttle, brake, gear

    def get_throttle_and_brake(
        self,
        current_location,
        current_speed,
        current_section,
        waypoints,
        corner,
        distanceToCorner,
    ):
        """
        Returns throttle and brake values based off the car's current location and the radius of the approaching turn
        """

        targetSpeed = self.get_target_speed(corner["radius"], current_section)
        speed_data = self.speed_for_turn(distanceToCorner, targetSpeed, current_speed)

        throttle, brake = self.speed_data_to_throttle_and_brake(speed_data)
        return throttle, brake

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        """Returns throttle and brake values given a SpeedData object

        Args:
            speed_data (SpeedData)

        Returns:
            tuple: throttle, brake
        """
        
        # self.dprint("dist=" + str(round(speed_data.distance_to_section)) + " cs=" + str(round(speed_data.current_speed, 2))
        #             + " ts= " + str(round(speed_data.target_speed_at_distance, 2))
        #             + " maxs= " + str(round(speed_data.recommended_speed_now, 2)) + " pcnt= " + str(round(percent_of_max, 2)))
        
        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now
        speed_change_per_tick = 2.5  # Speed decrease in kph per tick
        percent_change_per_tick = 0.075  # speed drop for one time-tick of braking
        true_percent_change_per_tick = round(
            speed_change_per_tick / (speed_data.current_speed + 0.001), 5
        )
        speed_up_threshold = 0.9
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        brake_threshold_multiplier = 1.0
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        speed_change = round(speed_data.current_speed - self.previous_speed, 3)

        # Distance needed to reach the target speed after breaking
        brakingDist = (
            (speed_data.recommended_speed_now / 3.6) ** 2
            - (speed_data.current_speed / 3.6) ** 2
        ) / (-2 * speed_change_per_tick / 3.6) / 3.5
        # print(f"\nBraking distance: {brakingDist:.2f}\nDistance to corner: {speed_data.distance_to_corner:.2f}\nRecommended speed: {speed_data.recommended_speed_now:.2f}\nTarget Speed: {speed_data.target_speed}")

        if brakingDist > speed_data.distance_to_corner:
            # Consider slowing down
            # if speed_data.current_speed > 200:  # Brake earlier at higher speeds
            #     brake_threshold_multiplier = 0.9

            if speed_data.current_speed > speed_data.target_speed:
                if self.brake_ticks > 0:
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: counter "
                        + str(self.brake_ticks)
                    )
                    return -1, 1

                # if speed is not decreasing fast, hit the brake.
                if self.brake_ticks <= 0 and speed_change < 3:
                    # start braking, and set for how many ticks to brake
                    self.brake_ticks = (
                        round(
                            (
                                speed_data.current_speed
                                - speed_data.target_speed
                            )
                            / (speed_change_per_tick)
                        )
                        + 2
                    )
                    # self.brake_ticks = 1, or (1 or 2 but not more)
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: initiate counter "
                        + str(self.brake_ticks)
                    )
                    return -1, 1

                else:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: throttle early1: sp_ch="
                        + str(percent_speed_change)
                    )
                    self.brake_ticks = 0  # done slowing down. clear brake_ticks
                    return 1, 0
            else:
                if speed_change >= 1.5:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: throttle early2: sp_ch="
                        + str(percent_speed_change)
                    )
                    self.brake_ticks = 0  # done slowing down. clear brake_ticks
                    return 1, 0

                throttle_to_maintain = self.get_throttle_to_maintain_speed(
                    speed_data.current_speed
                )

                if percent_of_max > 1.02 or percent_speed_change > (
                    -true_percent_change_per_tick / 2
                ):
                    self.dprint(
                        "tb: tick "
                        + str(self.tick_counter)
                        + " brake: throttle down: sp_ch="
                        + str(percent_speed_change)
                    )
                    return (
                        throttle_to_maintain * throttle_decrease_multiple,
                        0,
                    )  # coast, to slow down
                else:
                    # self.dprint("tb: tick " + str(self.tick_counter) + " brake: throttle maintain: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain, 0
        else:
            self.brake_ticks = 0  # done slowing down. clear brake_ticks
            # Speed up
            if speed_change >= 2.75:
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle: full speed drop: sp_ch="
                    + str(percent_speed_change)
                )
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle full: p_max="
                    + str(percent_of_max)
                )
                return 1, 0
            throttle_to_maintain = self.get_throttle_to_maintain_speed(
                speed_data.current_speed
            )
            if percent_of_max < 0.98 or true_percent_change_per_tick < -0.01:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle up: sp_ch="
                    + str(percent_speed_change)
                )
                return throttle_to_maintain * throttle_increase_multiple, 0
            else:
                self.dprint(
                    "tb: tick "
                    + str(self.tick_counter)
                    + " throttle maintain: sp_ch="
                    + str(percent_speed_change)
                )
                return throttle_to_maintain, 0

    # used to detect when speed is dropping due to brakes applied earlier. speed delta has a steep negative slope.
    def isSpeedDroppingFast(self, percent_change_per_tick: float, current_speed):
        """
        Detects if the speed of the car is dropping quickly.
        Returns true if the speed is dropping fast
        """
        percent_speed_change = (current_speed - self.previous_speed) / (
            self.previous_speed + 0.0001
        )  # avoid division by zero
        return percent_speed_change < (-percent_change_per_tick / 2)

    def get_throttle_to_maintain_speed(self, current_speed: float):
        """
        Returns a throttle value to maintain the current speed
        """
        throttle = 0.75 + current_speed / 500
        return throttle

    def speed_for_turn(
        self, distance: float, target_speed: float, current_speed: float
    ):
        """Generates a SpeedData object with the target speed for the far

        Args:
            distance (float): Distance from the start of the curve
            target_speed (float): Target speed of the curve
            current_speed (float): Current speed of the car

        Returns:
            SpeedData: A SpeedData object containing the distance to the corner, current speed, target speed, and max speed
        """
        # Takes in a target speed and distance and produces a speed that the car should target. Returns a SpeedData object

        d = (target_speed ** 2) / 300 + distance
        recommendedSpeed = (300 * d) ** 0.5
        return SpeedData(distance, current_speed, target_speed, recommendedSpeed)

    def get_next_interesting_waypoints(self, current_location, more_waypoints):
        """Returns a list of waypoints that are approximately as far as specified in intended_target_distance from the current location

        Args:
            current_location (roar_py_interface.RoarPyWaypoint): The current location of the car
            more_waypoints ([roar_py_interface.RoarPyWaypoint]): A list of waypoints

        Returns:
            [roar_py_interface.RoarPyWaypoint]: A list of waypoints within specified distances of the car
        """
        # Returns a list of waypoints that are approximately as far as the given in intended_target_distance from the current location

        # return a list of points with distances approximately as given
        # in intended_target_distance[] from the current location.
        points = []
        dist = []  # for debugging
        start = roar_py_interface.RoarPyWaypoint(
            current_location, np.ndarray([0, 0, 0]), 0.0
        )
        # start = self.agent.vehicle.transform
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            # print("start " + str(start) + "\n- - - - -\n")
            # print("end " + str(end) +     "\n- - - - -\n")
            curr_dist += distance_p_to_p(start, end)
            # curr_dist += start.location.distance(end.location)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        self.dprint("wp dist " + str(dist))
        return points

    def get_target_speed(self, radius: float, current_section: int):
        """Returns a target speed based on the radius of the turn and the section it is in

        Args:
            radius (float): The radius of the turn
            current_section (int): The current section of the track the car is in

        Returns:
            float: The maximum speed the car can go around the corner at
        """
        
        mu = 3

        # if radius >= self.max_radius:
        #     return self.max_speed

        # if current_section == 2:
        #     mu = 3.15
        # if current_section == 3:
        #     mu = 2
        if current_section in [4, 5]:
            mu = 4.5
        if current_section == 6:
            mu = 2.8
        if current_section == 9:
            mu = 4.5

        target_speed = math.sqrt(mu * 9.81 * radius) * 3.6 

        return max(
            20, min(target_speed, self.max_speed)
        )  # clamp between 20 and max_speed

    # def print_speed(
    #     self, text: str, s1: float, s2: float, s3: float, s4: float, curr_s: float
    # ):
    #     """
    #     Prints debug speed values
    #     """
    #     self.dprint(
    #         text
    #         + " s1= "
    #         + str(round(s1, 2))
    #         + " s2= "
    #         + str(round(s2, 2))
    #         + " s3= "
    #         + str(round(s3, 2))
    #         + " s4= "
    #         + str(round(s4, 2))
    #         + " cspeed= "
    #         + str(round(curr_s, 2))
    #     )

    # debug print
    def dprint(self, text):
        """
        Prints debug text
        """
        # print(text)
        # self.debug_strings.append(text)
