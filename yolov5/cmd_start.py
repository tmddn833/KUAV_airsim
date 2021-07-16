import setup_path
import airsim
from drone_control import euler_from_quaternion

import time
from threading import Thread
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
w_val= client.simGetCameraInfo("0").pose.orientation.w_val
x_val= client.simGetCameraInfo("0").pose.orientation.x_val
y_val= client.simGetCameraInfo("0").pose.orientation.y_val
z_val= client.simGetCameraInfo("0").pose.orientation.z_val

euler = euler_from_quaternion(x_val, y_val, z_val, w_val)

print(w_val, x_val, y_val, z_val)
print(euler)





import math
#
# #real world camera detection area : 대각선 길이
# cam_angle = 60/180 * math.pi #카메라 드론 각도
# xdFoV = 75/180 * math.pi #카메라 화각
# flying_angle = 5/180 * math.pi #비행각
#
# real_scale = 30 * (math.tan(xdFoV/2-math.pi/2+cam_angle+flying_angle) +
#                    math.tan(math.pi/2-(cam_angle+flying_angle)+xdFoV/2))
#
# print(real_scale)
#
# #horizontal distance between drone and person
# person_distance = 30*math.tan(math.pi/2-(cam_angle+flying_angle))
# print(person_distance)
#
# #display pixel : x_prime x y_prime 1280x720
# x_prime = 1280
# y_prime = 720
#
# #detected position
# #location = (x,y) = ((int(xyxy[2])+int(xyxy[0]))/2, (int(xyxy[3])-int(xyxy[1]))/2)
# location = (x,y) = (1000,500)
# print(location)
#
# #error from center
# error = [x_prime/2-x, y_prime-y]
# print(error)
#
# #scaler for real distance  #resolution of camera pixel : 1280x720
# pixel_x = 1280
# pixel_y = 720
# scaler_x = real_scale * math.cos(math.atan(pixel_y/pixel_x))/x_prime
# scaler_y = real_scale * math.sin(math.atan(pixel_y/pixel_x))/y_prime
# scaler = (scaler_x, scaler_y)
#
# #real_distance = error * scaler
# real_distance = [error[0] * scaler[0], error[1] * scaler[1]]
# print(real_distance)