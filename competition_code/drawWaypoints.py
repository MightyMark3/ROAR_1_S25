import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
import os
from progress.bar import IncrementalBar
import transforms3d as tr3d

baseSection = []
replacementSection = []
findTarget = False

print("\nLoading Waypoints\n")
waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
)
track = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(f"{os.path.dirname(__file__)}\\waypoints\\monzaOriginalWaypoints.npz")
)

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

# for i in additionalWaypoints:
#     plt.plot(i.location[0], i.location[1], "g^")

progressBar.finish()
print()
plt.show()
