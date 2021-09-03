import airsim
import math
from math import sqrt, acos, atan2, sin, cos
import time
import threading
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_location_metres, get_metres_location, euler_from_quaternion, quaternion_rotation_matrix, \
    get_intersections
from plot_results import plot_results
from shapely.geometry import LineString
from shapely.geometry import Point

NO_FLY_RADIUS = 10  # 10m


class MyMultirotorClient(airsim.MultirotorClient):
    def __init__(self, ip="", port=41451, timeout_value=3600,
                 default_gimbal_pitch=-math.pi / 4,
                 xdFoV=63 / 180 * math.pi,
                 hovering_altitude=-30,
                 velocity_gain=0.1,
                 track_target=False,
                 plot_traj=False
                 ):
        super(airsim.MultirotorClient, self).__init__(ip, port, timeout_value)
        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)
        self.takeoffAsync(timeout_sec=1)
        self.save_dir = None

        # Initial settings
        self.track_target = track_target
        self.hovering_altitude = hovering_altitude
        self.default_gimbal_pitch = default_gimbal_pitch
        self.following_distance = hovering_altitude / math.tan(default_gimbal_pitch)
        self.velocity_gain = velocity_gain
        self.xdFoV = xdFoV
        self.plot_traj = plot_traj

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
        # x,y # to estimate the location of human
        self.img_human_center = self.img_human_foot = (self.w / 2, self.h / 2)
        self.img_dx = 0  # camera_center ~ img_human_center error
        self.img_dy = 0
        self.human_detect = False

        # gimbal
        self.g_pitch_limit = [-math.pi / 2, 0]
        self.g_yaw_limit = [-math.pi, math.pi]

        # gps data
        self.start_gps = self.multirotor_state.gps_location
        self.human_lon_lat = None
        self.drone_lon_lat = [self.start_gps.longitude, self.start_gps.latitude]

        # mission mode
        self.mission_mode = "START"
        self.simPrintLogMessage("Mission Mode: ", self.mission_mode)
        self.start = self.multirotor_state.kinematics_estimated.position
        self.no_fly_center = [None, None]
        self.no_fly_center_gps = [None, None]
        self.AVOID_Tangent_flag = False
        self.FOLLOW_flag = True

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

    def mission_start(self, initial_point, no_fly_center, coordinate='XYZ', ):
        """
        :param initial_point: xy or gps(lat, lon) initial coordinate
        :param no_fly_center:no_fly zone center coordinate
        :param coordinate: 'XYZ' or 'GPS' coordinate option
        :return: go to mission starting place
        """
        # First takeoff
        if self.mission_mode != "START":
            return
        if no_fly_center:
            if coordinate == 'XYZ':
                self.no_fly_center = no_fly_center
                self.no_fly_center_gps = get_location_metres([self.start_gps.latitude, self.start_gps.longitude],
                                                             no_fly_center)
            else:  # gps
                self.no_fly_center = get_metres_location([self.start_gps.latitude, self.start_gps.longitude],
                                                         no_fly_center)
                self.no_fly_center_gps = no_fly_center
            print("No Fly Zone center in meters : ", self.no_fly_center)
            print("No Fly Zone center in GPS: ", self.no_fly_center_gps)

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
            print("Human Starting point in meters: ", initial_point)
            print("Human Starting point in GPS: ",
                  get_location_metres([self.start_gps.latitude, self.start_gps.longitude], initial_point))
            print("Drone mission starting point in meters :", [target_x, target_y])
            print("Drone mission starting GPS :",
                  get_location_metres([self.start_gps.latitude, self.start_gps.longitude], [target_x, target_y]))
        elif coordinate == 'GPS':
            D_lat_lon = [self.start_gps.latitude, self.start_gps.longitude]
            H_lat_lon = initial_point
            self.human_lon_lat = [initial_point[1], initial_point[0]]
            [H_x, H_y] = get_metres_location(D_lat_lon, H_lat_lon)
            real_dist = math.sqrt(H_x ** 2 + H_y ** 2)
            error_dist = real_dist - self.following_distance
            target_y = H_y * ((error_dist) / real_dist)
            target_x = H_x * ((error_dist) / real_dist)
            print("Human Starting point in meters: ", [H_x, H_y])
            print("Human Starting point in GPS: ", H_lat_lon)
            print("Drone mission starting point in meters :", [target_x, target_y])
            print("Drone mission starting GPS :",
                  get_location_metres(D_lat_lon, [target_x, target_y]))

        # Movement command
        target_yaw = math.atan2(H_y ,H_x)
        orientation_D = self.multirotor_state.kinematics_estimated.orientation
        orientation_G = self.camera_state.pose.orientation
        r, p, y = euler_from_quaternion(orientation_G)
        dr, dp, dy = euler_from_quaternion(orientation_D)
        self.g_roll, self.g_pitch, self.g_yaw = r - dr, self.default_gimbal_pitch - dp, 0
        self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                               airsim.to_quaternion(self.g_pitch, 0, 0)))
        print('move to initial point')
        self.moveToPositionAsync(target_x, target_y, self.hovering_altitude,
                                 yaw_mode=airsim.YawMode(False, target_yaw * 180 / math.pi),
                                 velocity=5).join()
        # self.moveToPositionAsync(0, 0, self.hovering_altitude, velocity=5).join()
        self.mission_mode = 'WAITING'
        self.simPrintLogMessage("Mission Mode: ", self.mission_mode)

    def drone_contol(self):
        self.adjust_target_gps()
        # self.adjust_gimbal_angle()
        if self.plot_traj:
            self.plot_trajectory()

    def load_sim_info(self):
        self.multirotor_state_temp = self.getMultirotorState()
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
        self.mission_mode = "FOLLOWING"
        self.simPrintLogMessage("Mission Mode: ", self.mission_mode)
        target_y = D_y + (H_y - D_y) * ((error_dist) / real_dist)
        target_x = D_x + (H_x - D_x) * ((error_dist) / real_dist)

        # AVOID1
        if (self.AVOID_Tangent_flag == False) and (NO_FLY_RADIUS + 3 >= math.sqrt(
                (H_x - self.no_fly_center[0]) ** 2 + (H_y - self.no_fly_center[1]) ** 2)):  # 사람 금지구역 in
            self.AVOID_Tangent_flag = True

        # 드론 & tangent_point distance <= 2 : 드론이 탄젠트 포인트에 들어옴
        elif (self.AVOID_Tangent_flag == True) and math.sqrt(
                    (self.T1[0] - self.tangent_tracking_point[0]) ** 2 + (self.T1[1] - self.tangent_tracking_point[1]) ** 2) <= 2:
            self.AVOID_Tangent_flag = False

        if self.AVOID_Tangent_flag == True:
            # circle_tangent_point
            b = sqrt((H_x - D_x) ** 2 + (H_y - D_y) ** 2)  # hypot() also works here
            th = acos((NO_FLY_RADIUS + 3) / b)  # angle theta
            d = atan2(H_y - D_y, H_x - D_x)  # direction angle of point P from C
            d1 = d + th  # direction angle of point T1 from C
            d2 = d - th  # direction angle of point T2 from C

            T1x = D_x + (NO_FLY_RADIUS + 3) * cos(d1)
            T1y = D_y + (NO_FLY_RADIUS + 3) * sin(d1)
            T2x = D_x + (NO_FLY_RADIUS + 3) * cos(d2)
            T2y = D_y + (NO_FLY_RADIUS + 3) * sin(d2)
            self.T1 = (T1x, T1y); self.T2 = (T2x, T2y)
            # line_circle_intersection
            H_p = Point(H_x, H_y)
            c = H_p.buffer(self.following_distance).boundary
            l = LineString([(D_x, D_y), self.T1])
            i = c.intersection(l)
            if len(i) == 2:
                length_1 = math.sqrt((D_x - i.geoms[0].coords[0][0]) ** 2 + (D_y - i.geoms[0].coords[0][1]) ** 2)
                length_2 = math.sqrt((D_x - i.geoms[1].coords[0][0]) ** 2 + (D_y - i.geoms[1].coords[0][1]) ** 2)
                tangent_tracking_point = i.geoms[0].coords[0] if length_1 <= length_2 else i.geoms[1].coords[0]
            else:
                tangent_tracking_point = i.geoms[0].coords[0]  # tuple has x , y
            target_x, target_y = tangent_tracking_point

        elif NO_FLY_RADIUS + 3 >= math.sqrt(
                (target_x - self.no_fly_center[0]) ** 2 + (target_y - self.no_fly_center[1]) ** 2):
            target_x, target_y = get_intersections([H_x, H_y], self.following_distance, self.no_fly_center,
                                                   NO_FLY_RADIUS + 3)[0]  # no fly radius # H : 현재 사람 위치

        dx, dy = target_x - D_x, target_y - D_y
        target_yaw = math.atan2((H_y - D_y),(H_x - D_x))
        if self.track_target:
            self.simPrintLogMessage("Mission Mode: ", self.mission_mode)
            self.moveByVelocityZAsync(self.velocity_gain * dx, self.velocity_gain * dy, self.hovering_altitude,
                                      duration=self.velocity_gain,
                                      yaw_mode=airsim.YawMode(False, target_yaw * 180 / math.pi))

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
            x_pgain = 0.00005
            self.g_pitch = max(min(self.g_pitch + self.img_dy * y_pgain, self.g_pitch_limit[1]), self.g_pitch_limit[0])
            self.g_yaw = max(min(self.g_yaw + self.img_dx * x_pgain, self.g_yaw_limit[1]), self.g_yaw_limit[0])
            # print(self.simGetCameraInfo("0"))
            self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                   airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))

    def plot_trajectory(self):
        # plot the traj of drone
        # self.past_point = self.multirotor_state.kinematics_estimated.position
        #
        # self.pose = self.simGetVehiclePose().position
        # self.simPlotLineStrip([airsim.Vector3r(self.past_point.x_val, self.past_point.y_val, self.past_point.z_val),
        #                        airsim.Vector3r(self.pose.x_val, self.pose.y_val, self.pose.z_val)], is_persistent=True)
        # self.past_point = self.pose
        try:
            target = self.drone_target_record[-1]
            self.simPlotPoints([airsim.Vector3r(target[0], target[1], self.hovering_altitude)], duration=4,
                               color_rgba=[1.0, 1.0, 1.0, 0.5])
        except:
            pass

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
        plot_results(self.save_dir, self.no_fly_center_gps)
