import airsim
import math
import time
import threading
import folium
from selenium import webdriver
import os
import cv2
import numpy as np


class MyMultirotorClient(airsim.MultirotorClient):
    def __init__(self, ip="", port=41451, timeout_value=3600,
                 default_gimbal_pitch=-math.pi / 4,
                 xdFoV=63 / 180 * math.pi,
                 hovering_altitude=-30,
                 velocity_gain=0.1,
                 track_target=False,
                 plot_threading = False,
                 plot_client = None):
        super(airsim.MultirotorClient, self).__init__(ip, port, timeout_value)
        self.confirmConnection()
        self.enableApiControl(True)
        self.save_dir = None

        # Initial settings
        self.track_target = track_target
        self.hovering_altitude = hovering_altitude
        self.default_gimbal_pitch = default_gimbal_pitch
        self.following_distance = hovering_altitude/math.tan(default_gimbal_pitch)
        self.velocity_gain = velocity_gain
        self.xdFoV = xdFoV

        # image
        rawImage = self.simGetImage("0", airsim.ImageType.Scene)
        cap = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        self.w = int(cap.shape[1])
        self.h = int(cap.shape[0])
        f = self.w / (2 * math.tan(self.xdFoV / 2))
        self.K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

        # gimbal
        self.simSetCameraPose("0",
                              airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(default_gimbal_pitch, 0, 0)))
        orientation_G = self.simGetCameraInfo("0").pose.orientation
        orientation_D = self.getMultirotorState().kinematics_estimated.orientation
        r, p, y = euler_from_quaternion(orientation_G)
        dr, dp, dy = euler_from_quaternion(orientation_D)
        self.g_roll, self.g_pitch, self.g_yaw = r - dr, p - dp, y - dy
        self.img_dx = 0  # camera_center ~ img_human_center error
        self.img_dy = 0
        self.g_pitch_limit = [-math.pi / 2, 0]
        self.g_yaw_limit = [-math.pi, math.pi]

        # x,y # to estimate the location of human
        self.img_human_center = self.img_human_foot = (self.w / 2, self.h / 2)
        self.human_detect = False
        self.start = self.getMultirotorState().kinematics_estimated.position
        self.gps_loop_time = time.time()
        
        # mission mode
        self.mission_mode = "Start"

        #plot the simulation
        if plot_threading:
            self.plot_client = plot_client
            self.thread1 = threading.Thread(target=self.plot_client.plot_traj, daemon=True)
            # self.thread2 = threading.Thread(target=self.connectGCS)
            self.thread1.start()
            # self.thread2.start()

    def mission_start(self, initial_point, coordinate = 'XYZ'):
        """
        :param initial_point: xy or gps(lon, lat) initial coordinate
        :param coordinate: 'XYZ' or 'GPS' coordinate option
        :return: go to mission starting place
        """
        if self.mission_mode !="Start":
            return

        if coordinate == 'XYZ':
            D_pose = self.getMultirotorState().kinematics_estimated.position
            D_y = D_pose.y_val
            D_x = D_pose.x_val
            H_x, H_y = initial_point
            real_dist = math.sqrt((D_x - H_x) ** 2 + (D_y - H_y) ** 2)
            error_dist = real_dist - self.following_distance
            target_y = D_y + (H_y - D_y) * ((error_dist) / real_dist)
            target_x = D_x + (H_x - D_x) * ((error_dist) / real_dist)
        elif coordinate == 'GPS':
            pass
        print(self.simGetObjectPose('NPC_3').position.x_val,self.simGetObjectPose('NPC_3').position.y_val)
        print(D_x, D_y)
        print(target_x, target_y)
        self.moveToPositionAsync(target_x, target_y, self.hovering_altitude, velocity=3, yaw_mode=airsim.YawMode(False, 0))
        self.g_yaw = math.atan((H_y - D_y)/(H_x - D_x))
        self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                               airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))


    def adjust_target_gps(self):
        #trace by image detected point
        if self.human_detect is False:
            return

        # To get the absolute coordinate of human, Find the rotation matrix
        # O : origin D : drone, G : gimbal
        # Drone coord : front = North -> x, right = East -> y, Downward -> z
        D_orientation = self.getMultirotorState().kinematics_estimated.orientation
        G_orientation = self.simGetCameraInfo("0").pose.orientation
        OR_D = quaternion_rotation_matrix(D_orientation)

        # Correct the rotation matrix -> G_orientation is absolute coordinate rotation
        # DR_G = quaternion_rotation_matrix(G_orientation)
        # OR_G = np.matmul(OR_D,DR_G)
        OR_G = quaternion_rotation_matrix(G_orientation)

        D_pose = self.getMultirotorState().kinematics_estimated.position
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

        # # if GPS, use haversine!!!
        # realdist = haversine(, gps_drone, unit='m')
        real_dist = math.sqrt((D_x - H_x) ** 2 + (D_y - H_y) ** 2)
        error_dist = real_dist - self.following_distance

        target_y = D_y + (H_y - D_y) * ((error_dist) / real_dist)
        target_x = D_x + (H_x - D_x) * ((error_dist) / real_dist)
        dx, dy = target_x - D_x, target_y - D_y

        # To check the estimated value
        record1 = open(str(self.save_dir / 'estimated_personcoord_record.txt'), 'a')
        record1.write(str((H_x, H_y)) + '\n')
        record1.close()

        record2 = open(str(self.save_dir / 'actual_personcoord_record.txt'), 'a')
        record = open(str(self.save_dir / 'error_distance_real.txt'), 'a')
        x_real = self.simGetObjectPose("NPC_3").position.x_val
        y_real = self.simGetObjectPose("NPC_3").position.y_val
        error_dist_real = math.sqrt((x_real-D_x)**2 + (y_real-D_y)**2) - self.following_distance
        record2.write(str((x_real, y_real)) + '\n')
        record.write(str(error_dist_real) + '\n')
        record2.close()
        record.close()

        record3 = open(str(self.save_dir / 'error_distance_est.txt'), 'a')
        record3.write(str(error_dist) + '\n')
        record3.close()

        record4 = open(str(self.save_dir / 'drone_coord_record.txt'), 'a')
        record4.write(str((D_x, D_y)) + '\n')
        record4.close()

        if self.track_target:
            self.moveByVelocityZAsync(self.velocity_gain * dx, self.velocity_gain * dy, self.hovering_altitude,
                                      duration=time.time() - self.gps_loop_time)
            self.gps_loop_time = time.time()
            # self.moveToPositionAsync(target_x, target_y, self.hovering_altitude, velocity=error_dist*self.velocity_gain,
            #                          timeout_sec=0.1)
            record4 = open(str(self.save_dir / 'drone_target_record.txt'), 'a')
            record4.write(str((target_x, target_y)) + '\n')
            record4.close()


    def adjust_target_gps_old(self):
        # trace by location of center of the camera
        # To get the absolute coordinate of human, Find the rotation matrix
        # O : origin D : drone, G : gimbal
        D_orientation = self.getMultirotorState().kinematics_estimated.orientation
        G_orientation = self.simGetCameraInfo("0").pose.orientation
        OR_D = quaternion_rotation_matrix(D_orientation)
        DR_G = quaternion_rotation_matrix(G_orientation)
        OR_G = np.matmul(OR_D, DR_G)
        D_pose = self.getMultirotorState().kinematics_estimated.position
        D_z = D_pose.z_val
        D_x = D_pose.x_val
        D_y = D_pose.y_val
        d1 = -D_z / OR_G[2, 0]
        H_x = D_x + OR_G[0, 0] * d1
        H_y = D_y + OR_G[1, 0] * d1

        # # if GPS, use haversine!!!
        # realdist = haversine(, gps_drone, unit='m')
        real_dist = math.sqrt((D_x - H_x) ** 2 + (D_y - H_y) ** 2)
        error_dist = real_dist - self.following_distance

        target_y = D_y + (H_y - D_y) * ((error_dist) / real_dist)
        target_x = D_x + (H_x - D_x) * ((error_dist) / real_dist)

        # To check the estimated value
        record1 = open('recordings/estimated_personcoord_record.txt', 'a')
        record1.write(str((H_x, H_y)) + '\n')
        record1.close()

        record2 = open('recordings/actual_personcoord_record.txt', 'a')
        x_real = self.simGetObjectPose("NPC_2").position.x_val
        y_real = self.simGetObjectPose("NPC_2").position.y_val
        record2.write(str((x_real, y_real)) + '\n')
        record2.close()

        record3 = open('recordings/error_distance.txt', 'a')
        record3.write(str(error_dist) + '\n')
        record3.close()

        record4 = open('recordings/drone_coord_record.txt', 'a')
        record4.write(str((D_x, D_y)) + '\n')
        record4.close()

        dx, dy = target_x - D_x, target_y - D_y

        if self.track_target:
            print((dx, dy))
            self.moveByVelocityZAsync(self.velocity_gain * dx, self.velocity_gain * dy, self.hovering_altitude,
                                      duration=time.time() - self.gps_loop_time)
            self.gps_loop_time = time.time()
            # self.moveToPositionAsync(target_x, target_y, self.hovering_altitude, velocity=error_dist*self.velocity_gain,
            #                          timeout_sec=0.1)
            record4 = open('recordings/drone_target_record.txt', 'a')
            record4.write(str((target_x, target_y)) + '\n')
            record4.close()

        # gps_drone = (37.582049, 127.026440)  # 드론 좌표 (위도(lat), 경도(lon))
        # gps_person = (37.582274, 127.026043)  # 사람 좌표

        # # 드론,사람 사이거리
        # gps_drone_lat = D_y
        # gps_drone_lon = D_x
        # gps_person_lat = H_y
        # gps_person_lon = H_x
        # gps_target_lat = gps_drone_lat + (gps_person_lat - gps_drone_lat) * ((result - 15) / result)
        # gps_target_lon = gps_drone_lon + (gps_person_lon - gps_drone_lon) * ((result - 15) / result)
        # gps_target = (gps_target_lat, gps_target_lon)  # 타겟 좌표

    def adjust_gimbal_angle(self):
        '''
        function to reflex the human point and adjust the gimbal's roll pitch yaw(temporal)
        :param human_center: human's (x,y)
        :param camera_center: camera's center (x,y)
        :return: void
        '''

        if self.img_human_center == (0, 0) or self.human_detect is False:
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
            self.img_dx = self.img_human_foot[0] - self.w / 2
            self.img_dy = self.img_human_foot[1] - self.h / 2
            y_pgain = -0.00015
            x_pgain = 0.00015
            self.g_pitch = max(min(self.g_pitch + self.img_dy * y_pgain, self.g_pitch_limit[1]), self.g_pitch_limit[0])
            self.g_yaw = max(min(self.g_yaw + self.img_dx * x_pgain, self.g_yaw_limit[1]), self.g_yaw_limit[0])
            # print(self.simGetCameraInfo("0"))
            self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                   airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))

    def plot_traj(self):
        past_point = self.simGetVehiclePose().position

        while True:
            euler = euler_from_quaternion(self.simGetCameraInfo("0").pose.orientation)

            # print(w_val, x_val, y_val, z_val)
            # print(euler)

            gps = self.getMultirotorState().gps_location
            alt = gps.altitude
            lon = gps.longitude
            lat = gps.latitude

            # print(alt, lon, lat)
            # plt.scatter(lon, lat)
            # plt.pause(0.001)
            pose = self.simGetVehiclePose().position
            self.simPlotLineStrip([airsim.Vector3r(past_point.x_val, past_point.y_val, past_point.z_val),
                                     airsim.Vector3r(pose.x_val, pose.y_val, pose.z_val)], is_persistent=True)
            past_point = pose

def get_location_metres(original_location, dxdy):
    """
    Returns a Location object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned Location has the same `alt and `is_relative` values
    as `original_location`.
    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0  # Radius of "spherical" earth
    # Coordinate offsets in radians
    dLat = dxdy[0] / earth_radius * 180 / math.pi
    dLon = dxdy[1] / (earth_radius * math.cos(math.pi * original_location[0] / 180))* 180 / math.pi

    # New position in decimal degrees
    newlat = original_location[0] + dLat
    newlon = original_location[1] + dLon
    return (newlat, newlon)

def euler_from_quaternion(orientation):
    """
    :param orientation = client.simGetMultiRotorState().pose.orientation

    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    # Note that the return value is in PRY order NOT RPY
    return roll_x, pitch_y, yaw_z  # in radians


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def quaternion_rotation_matrix(orientation):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    x, y, z, w = orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val
    q0, q1, q2, q3 = w, x, y, z
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

