import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
import json
import os
from progress.bar import IncrementalBar
import transforms3d as tr3d
import plotly.express as px
import pandas as pd

data = json.load(open(f"{os.path.dirname(__file__)}\\debugData\\debugData.json"))
lapMarkers = ["^", ".", "s"]
totalSpeedChangeWhileBraking = 0
numBrakeVals = 0
use2D = False


def distanceToWaypoint(currentWaypoint, firstWaypoint):
    return np.linalg.norm(currentWaypoint[:2] - firstWaypoint[:2])


print("\nLoading Waypoints\n")

track = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(f"{os.path.dirname(__file__)}\\waypoints\\Monza Original Waypoints.npz")
)

totalPoints = len(data) + len(track)
progressBar = IncrementalBar("Plotting points", max=totalPoints)

if use2D:
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
            width=0.25,
            color="r",
        )
        progressBar.next()

    for i in data:
        colorVal = (0, 0, 0)
        brakeVal = data[i]["brake"]
        throttleVal = data[i]["throttle"]

        if brakeVal > 0:
            colorVal = (brakeVal**2, 0, 0)
            totalSpeedChangeWhileBraking += prevData["speed"] - data[i]["speed"]
            numBrakeVals += 1
        else:
            colorVal = (0, throttleVal**2, 0)

        x = data[i]["loc"][0]
        y = data[i]["loc"][1]

        prevData = data[i]

        plt.plot(x, y, lapMarkers[data[i]["lap"] - 1], color=colorVal)
        progressBar.next()
else:
    points = []
    for i in data:
        # Add a list of x, y, speed, and throttle/brake values to points
        points.append(
            [
                data[i]["loc"][0],
                data[i]["loc"][1],
                data[i]["speed"],
                data[i]["throttle"] - data[i]["brake"],
            ]
        )
        # if data[i]["brake"] != 0:
        #     numBrakeVals += 1
        #     totalSpeedChangeWhileBraking = data[i]["speed"] - prevData["speed"]
        # prevData = data[i]
        progressBar.next()

    # Convert our list of points into a Pandas DataFrame to use with Plotly
    df = pd.DataFrame(points, columns=["x", "y", "z", "throttle/brake"])

    # Plot the DataFrame
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="throttle/brake",
        color_continuous_scale=[[0, "rgb(225,0,0)"], [1, "rgb(0,225,0)"]],
        range_color=[-1, 1],
        labels={"x": "X", "y": "Y", "z": "Speed (kph)"},
    )
    fig.show()

progressBar.finish()
print()
# print(f"Average speed change while braking: {totalSpeedChangeWhileBraking / numBrakeVals}\n")
plt.show()
