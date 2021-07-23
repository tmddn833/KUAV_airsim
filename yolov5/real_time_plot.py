import matplotlib.pyplot as plt
import airsim
from drone_control import euler_from_quaternion

import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
figure = plt.figure()
while True:
    plt.figure(1)
    plt.plot(euler_from_quaternion(client.getMultirotorState().kinematics_estimated.orientation))
    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.figure(2)
    x = client.getMultirotorState().kinematics_estimated.position.x_val
    y = client.getMultirotorState().kinematics_estimated.position.y_val
    z = client.getMultirotorState().kinematics_estimated.position.z_val
    plt.stem([x,y,z])
    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.pause(0.1)
    plt.clf()

    plt.figure(1)
    plt.plot((client.getMultirotorState().kinematics_estimated.orientation))
    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.pause(0.1)
    plt.clf()
