import airsim
import math
import time
import threading
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_location_metres, get_metres_location, euler_from_quaternion, quaternion_rotation_matrix
from plot_results import plot_results


class MyMultirotorClient(airsim.MultirotorClient):
    def __init__(self, ip="", port=41451, timeout_value=3600,
                 default_gimbal_pitch=-math.pi / 4,
                 xdFoV=63 / 180 * math.pi,
                 hovering_altitude=-30,
                 velocity_gain=0.1,
                 track_target=False,
                 plot_threading=False
                 ):
        super(airsim.MultirotorClient, self).__init__(ip, port, timeout_value)
        self.confirmConnection()
        self.enableApiControl(True)
        self.save_dir = None

        # Initial settings
        self.track_target = track_target
        self.hovering_altitude = hovering_altitude
        self.default_gimbal_pitch = default_gimbal_pitch
        self.following_distance = hovering_altitude / math.tan(default_gimbal_pitch)
        self.velocity_gain = velocity_gain
        self.xdFoV = xdFoV
        self.plot_threading = plot_threading

        # get starting info
        self.multirotor_state = self.getMultirotorState()
        self.camera_state = self.simGetCameraInfo("0")
        rawImage = self.simGetImage("0", airsim.ImageType.Scene)

        # image
        cap = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        self.w = int(cap.shape[1])
        self.h = int(cap.shape[0])
        f = self.w / (2 * math.tan(self.xdFoV / 2))
        self.K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

        # gimbal
        self.g_pitch_limit = [-math.pi / 2, 0]
        self.g_yaw_limit = [-math.pi, math.pi]

        # x,y # to estimate the location of human
        self.img_human_center = self.img_human_foot = (self.w / 2, self.h / 2)
        self.img_dx = 0  # camera_center ~ img_human_center error
        self.img_dy = 0

        self.human_detect = False
        self.start = self.multirotor_state.kinematics_estimated.position

        # gps data
        self.start_gps = self.multirotor_state.gps_location
        self.human_lon_lat = None
        self.drone_lon_lat = [self.start_gps.longitude, self.start_gps.latitude]

        # mission mode
        self.mission_mode = "Start"

        # recordings
        self.estimated_personcoord_record = []
        self.actual_personcoord_record = []
        self.error_distance_real_record = []
        self.error_distance_est_record = []
        self.drone_coord_record = []
        self.drone_target_record = []
        self.drone_lon_lat_record = []
        self.human_lon_lat_record = []
        self.human_lon_lat_real_record = []

    def mission_start(self, initial_point, coordinate='XYZ'):
        """
        :param initial_point: xy or gps(lon, lat) initial coordinate
        :param coordinate: 'XYZ' or 'GPS' coordinate option
        :return: go to mission starting place
        """
        # First takeoff
        if self.mission_mode != "Start":
            return

        if coordinate == 'XYZ':
            # D_pose = self.multirotor_state.kinematics_estimated.position
            D_y = self.start.y_val
            D_x = self.start.x_val
            H_x, H_y = initial_point
            temp = get_location_metres([self.start_gps.latitude, self.start_gps.longitude], [H_x, H_y])
            self.human_lon_lat = [temp[1], temp[0]]
            real_dist = math.sqrt((D_x - H_x) ** 2 + (D_y - H_y) ** 2)
            error_dist = real_dist - self.following_distance
            target_y = D_y + (H_y - D_y) * ((error_dist) / real_dist)
            target_x = D_x + (H_x - D_x) * ((error_dist) / real_dist)
        elif coordinate == 'GPS':
            pass

        print("Human Starting point in meters: ", initial_point)
        print("Human Starting point in GPS: ",
              get_location_metres([self.start_gps.latitude, self.start_gps.longitude], initial_point))
        print("Drone mission starting point in meters :", [target_x, target_y])
        print("Drone mission starting GPS :",
              get_location_metres([self.start_gps.latitude, self.start_gps.longitude], [target_x, target_y]))

        # Movement command
        target_yaw = math.atan((H_y - D_y) / (H_x - D_x))
        orientation_D = self.multirotor_state.kinematics_estimated.orientation
        orientation_G = self.camera_state.pose.orientation
        r, p, y = euler_from_quaternion(orientation_G)
        dr, dp, dy = euler_from_quaternion(orientation_D)
        self.g_roll, self.g_pitch, self.g_yaw = r - dr, self.default_gimbal_pitch - dp, target_yaw - dy
        self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                               airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))
        print('move to initial point')
        self.moveToPositionAsync(target_x, target_y, self.hovering_altitude, velocity=5).join()
        # self.moveToPositionAsync(0, 0, self.hovering_altitude, velocity=5).join()

    def drone_contol(self):
        self.adjust_target_gps()
        self.adjust_gimbal_angle()
        if self.plot_threading:
            self.plot_traj()

    def load_sim_info(self):
        self.multirotor_state_temp= self.getMultirotorState()
        # self.camera_state_temp = self.simGetCameraInfo("0")

    def read_sim_info(self):
        self.multirotor_state = self.multirotor_state_temp
        # self.camera_state = self.camera_state_temp

    def adjust_target_gps(self):
        # trace by image detected point

        if self.human_detect is False:
            return
        # To get the absolute coordinate of human, Find the rotation matrix
        # O : origin D : drone, G : gimbal
        # Drone coord : front = North -> x, right = East -> y, Downward -> z
        D_orientation = self.multirotor_state.kinematics_estimated.orientation
        # G_orientation = self.camera_state.pose.orientation
        OR_D = quaternion_rotation_matrix(D_orientation)
        DR_G = quaternion_rotation_matrix(airsim.to_quaternion(self.g_pitch, self.g_roll, self.g_yaw))

        # Correct the rotation matrix -> G_orientation is absolute coordinate rotation
        # DR_G = quaternion_rotation_matrix(G_orientation)
        OR_G = np.matmul(OR_D, DR_G)
        # OR_G = quaternion_rotation_matrix(G_orientation)
        D_pose = self.multirotor_state.kinematics_estimated.position
        D_z = D_pose.z_val
        D_y = D_pose.y_val
        D_x = D_pose.x_val
        T = np.array([[D_x], [D_y], [D_z]])
        # Make front -> z Left -> x below -> y on the direction the camera is looking
        # this is transformation to camera coordinate
        R_tran = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        OR_C = np.hstack((OR_G, T))
        OR_C = np.vstack((OR_C, np.array([[0, 0, 0, 1]])))
        OR_C = np.matmul(OR_C, R_tran)
        CR_O = np.linalg.inv(OR_C)
        A = np.dstack((CR_O[0:3, 0], CR_O[0:3, 1], CR_O[0:3, 3]))
        A = np.linalg.inv(A)
        xy_est = np.matmul(np.matmul(A, self.K_inv), np.array([self.img_human_foot[0], self.img_human_foot[1], 1]))
        xy_est = xy_est[0]
        xy_est = xy_est / xy_est[2]
        H_x, H_y = (xy_est[0], xy_est[1])
        real_dist = math.sqrt((D_x - H_x) ** 2 + (D_y - H_y) ** 2)
        error_dist = real_dist - self.following_distance

        target_y = D_y + (H_y - D_y) * ((error_dist) / real_dist)
        target_x = D_x + (H_x - D_x) * ((error_dist) / real_dist)
        dx, dy = target_x - D_x, target_y - D_y
        self.velocity_gain * dx
        if self.track_target:
            self.moveByVelocityZAsync(self.velocity_gain * dx, self.velocity_gain * dy, self.hovering_altitude, duration=self.velocity_gain)

        # recording
        human_pose = self.simGetObjectPose("NPC_3")
        x_real = human_pose.position.x_val
        y_real = human_pose.position.y_val
        drone_GPS = self.multirotor_state.gps_location
        self.drone_lon_lat = [drone_GPS.longitude, drone_GPS.latitude]
        temp = get_location_metres([self.start_gps.latitude, self.start_gps.longitude], [H_x, H_y])
        self.human_lon_lat = [temp[1], temp[0]]

        error_dist_real = math.sqrt((x_real - D_x) ** 2 + (y_real - D_y) ** 2) - self.following_distance
        self.estimated_personcoord_record.append([H_x, H_y])
        self.actual_personcoord_record.append([x_real, y_real])
        self.error_distance_real_record.append(error_dist_real)
        self.error_distance_est_record.append(error_dist)
        self.drone_coord_record.append([D_x, D_y])
        self.drone_lon_lat_record.append(self.drone_lon_lat)
        self.human_lon_lat_record.append(self.human_lon_lat)
        human_gps = get_location_metres([self.start_gps.latitude, self.start_gps.longitude], [x_real, y_real])
        self.human_lon_lat_real_record.append([human_gps[1], human_gps[0]])
        self.drone_target_record.append([target_x, target_y])

    def adjust_gimbal_angle(self):
        '''
        function to reflex the human point and adjust the gimbal's roll pitch yaw(temporal)
        '''

        if self.human_detect is False:
            y_ratio = 0.1
            x_ratio = 0.1
            self.img_dx -= self.img_dx * x_ratio
            self.img_dy -= self.img_dy * y_ratio
            y_pgain = -0.000153
            x_pgain = 0.000153
            self.g_pitch = max(min(self.g_pitch + self.img_dy * y_pgain, self.g_pitch_limit[1]), self.g_pitch_limit[0])
            self.g_yaw = max(min(self.g_yaw + self.img_dx * x_pgain, self.g_yaw_limit[1]), self.g_yaw_limit[0])
            # print(self.simGetCameraInfo("0"))
            self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                   airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))
        else:
            y_pgain = -0.00005
            x_pgain = 0.00015
            self.g_pitch = max(min(self.g_pitch + self.img_dy * y_pgain, self.g_pitch_limit[1]), self.g_pitch_limit[0])
            self.g_yaw = max(min(self.g_yaw + self.img_dx * x_pgain, self.g_yaw_limit[1]), self.g_yaw_limit[0])
            # print(self.simGetCameraInfo("0"))
            self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                   airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))

    def plot_traj(self):
        self.past_point = self.multirotor_state.kinematics_estimated.position

        self.pose = self.simGetVehiclePose().position
        self.simPlotLineStrip([airsim.Vector3r(self.past_point.x_val, self.past_point.y_val, self.past_point.z_val),
                               airsim.Vector3r(self.pose.x_val, self.pose.y_val, self.pose.z_val)], is_persistent=True)
        self.past_point = self.pose

    def live_plot(self):
        while True:
            gps = self.drone_lon_lat
            plt.scatter(gps[0], gps[1])
            plt.pause(0.001)

    def dataplot(self):
        df = pd.DataFrame(dtype=object)
        if self.track_target:
            df['drone_target_record'] = self.drone_target_record
        df['estimated_personcoord_record'] = self.estimated_personcoord_record
        df['actual_personcoord_record'] = self.actual_personcoord_record
        df['error_distance_real_record'] = self.error_distance_real_record
        df['error_distance_est_record'] = self.error_distance_est_record
        df['drone_coord_record'] = self.drone_coord_record
        df['drone_lon_lat_record'] = self.drone_lon_lat_record
        df['human_lon_lat_record'] = self.human_lon_lat_record
        df['human_lon_lat_real_record'] = self.human_lon_lat_real_record
        df.to_csv(self.save_dir / "simulation_data.csv")
        plot_results(self.save_dir)
