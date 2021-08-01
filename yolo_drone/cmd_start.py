import airsim
from drone_control import euler_from_quaternion

import time
import matplotlib.pyplot as plt
from threading import Thread
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
# client.enableApiControl(True)
# client.armDisarm(True)

start = client.getMultirotorState().gps_location

past_point = client.simGetVehiclePose().position

while True:
    euler = euler_from_quaternion(client.simGetCameraInfo("0").pose.orientation)

    # print(w_val, x_val, y_val, z_val)
    # print(euler)

    gps = client.getMultirotorState().gps_location
    alt = gps.altitude
    lon = gps.longitude
    lat = gps.latitude

    # print(alt, lon, lat)
    plt.scatter(lon, lat)
    plt.pause(0.001)
    pose = client.simGetVehiclePose().position
    client.simPlotLineStrip([airsim.Vector3r(past_point.x_val, past_point.y_val, past_point.z_val),
                             airsim.Vector3r(pose.x_val, pose.y_val, pose.z_val)], is_persistent=True)
    past_point = pose
    print(pose.x_val, pose.y_val, pose.z_val)