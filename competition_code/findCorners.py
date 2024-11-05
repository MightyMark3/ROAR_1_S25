import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
import os
from progress.bar import IncrementalBar
import transforms3d as tr3d
import math

def get_radius(wp: [roar_py_interface.RoarPyWaypoint]):
    """Returns the radius of a curve given 3 waypoints using the Menger Curvature Formula

    Args:
        wp ([roar_py_interface.RoarPyWaypoint]): A list of 3 RoarPyWaypoints

    Returns:
        float: The radius of the curve made by the 3 given waypoints
    """

    point1 = (wp[0].location[0], wp[0].location[1])
    point2 = (wp[1].location[0], wp[1].location[1])
    point3 = (wp[2].location[0], wp[2].location[1])

    # Calculating length of all three sides
    len_side_1 = round(math.dist(point1, point2), 3)
    len_side_2 = round(math.dist(point2, point3), 3)
    len_side_3 = round(math.dist(point1, point3), 3)

    small_num = 5

    if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
        return 1000000
    # sp is semi-perimeter
    sp = (len_side_1 + len_side_2 + len_side_3) / 2

    # Calculating area using Herons formula
    area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
    if area_squared < small_num:
        return 1000000
    # Calculating curvature using Menger curvature formula
    radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))

    return radius

baseSection = []
replacementSection = []
findTarget = False
corners = []

print("\nLoading Waypoints\n")
waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
)
track = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(f"{os.path.dirname(__file__)}\\waypoints\\Monza Original Waypoints.npz")
)

curAngle = track[0].roll_pitch_yaw[2]
angleDiffForCorner = 0.2
radForCorner = 100
isCorner = False
cornerStartIndex = None

for i in range(len(track) + 5):
    # smallRad = get_radius([track[i], track[(i + 4) % len(track)], track[(i + 8) % len(track)]])
    # medRad = get_radius([track[i], track[(i + 8) % len(track)], track[(i + 16) % len(track)]])
    # bigRad = get_radius([track[i], track[(i + 16) % len(track)], track[(i + 32) % len(track)]])
    # curRad = min(smallRad, medRad, bigRad)
    
    # if curRad > radForCorner and curRad < 500000 and not isCorner:
    #     cornerStart = track[i]
    #     isCorner = True
    # elif (curRad < radForCorner / 2 or curRad > 500000) and isCorner:
    #     isCorner = False
    #     cornerEnd = track[i]
    #     corners.append([cornerStart, cornerEnd])
    
    farAngleDiff = abs(curAngle - track[(i + 8) % len(track)].roll_pitch_yaw[2])
    shortAngleDiff = abs(curAngle - track[(i + 4) % len(track)].roll_pitch_yaw[2])
    if (farAngleDiff > angleDiffForCorner or (shortAngleDiff > angleDiffForCorner and farAngleDiff < 0.01)) and not isCorner:
        cornerStart = track[i % len(track)]
        cornerStartIndex = i
        isCorner = True
    elif farAngleDiff < 0.01 and isCorner:
        isCorner = False
        if i - cornerStartIndex > 10:
            cornerEnd = track[(i + 5) % len(track)]
            corners.append([cornerStart, cornerEnd])
    curAngle = track[i % len(track)].roll_pitch_yaw[2]

# print(corners)

totalPoints = len(waypoints) + len(track)
progressBar = IncrementalBar("Plotting points", max=totalPoints)

plt.figure(figsize=(11, 11))
plt.axis((-1100, 1100, -1100, 1100))
plt.tight_layout()

for waypoint in track[:] if track is not None else []:
    rep_line = waypoint.line_representation
    rep_line = np.asarray(rep_line)
    waypoint_heading = tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array(
        [1, 0, 0]
    )
    plt.plot(rep_line[:, 0], rep_line[:, 1], "k", linewidth=2)
    plt.arrow(
        waypoint.location[0],
        waypoint.location[1],
        waypoint_heading[0] * 1,
        waypoint_heading[1] * 1,
        width=0.5,
        color="r",
    )
    progressBar.next()

for i in waypoints:
    plt.plot(i.location[0], i.location[1], "ro")
    progressBar.next()

# print(corners)
for i in corners:
    # Start of corner
    plt.plot(i[0].location[0], i[0].location[1], "yo")
    
    # End of corner
    plt.plot(i[1].location[0], i[1].location[1], "go")
    progressBar.next()

# for i in additionalWaypoints:
#     plt.plot(i.location[0], i.location[1], "g^")

progressBar.finish()
print()
plt.show()
