import setup_path
import airsim

import sys
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.moveByMotorPWMsAsync(1,1,1,1,10).join()
client.moveByMotorPWMsAsync(1,0,1,0,10)