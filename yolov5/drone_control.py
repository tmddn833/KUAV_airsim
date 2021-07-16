import airsim
import math

class MyMultirotorClient(airsim.MultirotorClient):
    def __init__(self, ip = "", port = 41451, timeout_value = 3600, default_gimbal_pitch = -math.pi/4,
                 xdFoV = 63 / 180 * math.pi):
        super(airsim.MultirotorClient, self).__init__(ip, port, timeout_value)
        self.confirmConnection()
        self.enableApiControl(True)

        # gimbal
        self.default_gimbal_pitch = default_gimbal_pitch
        self.simSetCameraPose("0",airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(default_gimbal_pitch, 0, 0)))
        orientation = self.simGetCameraInfo("0").pose.orientation
        self.g_pitch, self.g_roll, self.g_yaw = euler_from_quaternion(orientation)
        self.img_dx = 0  # camera_center ~ img_human_center error
        self.img_dy = 0
        self.g_pitch_limit = [-math.pi/2, 0]
        self.g_yaw_limit = [-math.pi, math.pi]

        self.img_human_center = (0, 0)  # x,y
        self.human_detect = False
        self.start = self.getMultirotorState().kinematics_estimated.position

        self.xdFoV = xdFoV


    def adjust_target_gps(self, frame):
        # cam_angle = 60 / 180 * math.pi  # 카메라 드론 각도 -> self.default_gimbal_pitch
        # xdFoV = 75 / 180 * math.pi  # 카메라 화각 -> self.xdFoV
        flying_orientation = self.getMultirotorState().kinematics_estimated.orientation
        flying_angles = euler_from_quaternion(flying_orientation)
        flying_angle = -flying_angles[0] # (-) positive pitch -> look down, [0] PRY order
        z = -self.getMultirotorState().kinematics_estimated.position.z_val

        # real_scale ground distance
        real_scale = z * (math.tan(self.xdFoV / 2 - math.pi / 2 + - self.default_gimbal_pitch + flying_angle) +
                           math.tan(math.pi / 2 - (-self.default_gimbal_pitch + flying_angle) + self.xdFoV / 2))



        # display pixel : x_prime * y_prime 1280x720
        x_prime, y_prime = frame

        # error from center
        error = (self.img_dx, self.img_dy)
        print("error = ", error)

        # scaler real distance from resolution of camera pixel (dimention : distance/degree)
        scaler_x = real_scale * math.cos(math.atan(y_prime / x_prime)) / x_prime
        scaler_y = real_scale * math.sin(math.atan(y_prime / x_prime)) / y_prime

        # real_distance = error * scaler
        error_real_distance_coord = [error[0] * scaler_x, error[1] * scaler_y]
        print("error_real_distance_coord =", error_real_distance_coord)

        # distance : sqrt
        error_real_distance = math.sqrt(error_real_distance_coord[0] ** 2 + error_real_distance_coord[1] ** 2)
        print("error_real_distance =", error_real_distance)

        # angle
        if error_real_distance_coord[0] > 0 and error_real_distance_coord[1] > 0:  # 1사분면
            theta = math.atan(error_real_distance_coord[0] / error_real_distance_coord[1]) * (
                    360 / (2 * math.pi))
        elif error_real_distance_coord[0] < 0 and error_real_distance_coord[1] > 0:  # 2사분면
            theta = 360 + math.atan(error_real_distance_coord[0] / error_real_distance_coord[1]) * (
                    360 / (2 * math.pi))
        elif error_real_distance_coord[0] < 0 and error_real_distance_coord[1] < 0:  # 3사분면
            theta = math.atan(error_real_distance_coord[0] / error_real_distance_coord[1]) * (
                    360 / (2 * math.pi)) + 180
        elif error_real_distance_coord[0] > 0 and error_real_distance_coord[1] < 0:  # 4사분면
            theta = 180 + math.atan(error_real_distance_coord[0] / error_real_distance_coord[1]) * (
                    360 / (2 * math.pi))

        print("theta =", theta)

        # adjustment좌표 / 각조절 : -theta-90
        adj_x = error_real_distance * math.cos((-theta - 90) * (2 * math.pi) / 360)
        adj_y = error_real_distance * math.sin((-theta - 90) * (2 * math.pi) / 360)

        earth_radius = 6378137.0  # Radius of "spherical" earth

        # Coordinate offsets in radians
        dLat = adj_x / earth_radius
        dLon = adj_y / (earth_radius * math.cos(math.pi * y_old / 180))

        # 조정값
        x_new = x_old + dLat
        y_new = y_old + dLon

        point_new = (x_new, y_new)
        print("new_GPS :", point_new)

        x_old = x_new
        y_old = y_new

        ################################################################

    def adjust_gimbal_angle(self, camera_center):
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
            y_pgain = -0.00015
            x_pgain = 0.00015
            self.g_pitch = max(min(self.g_pitch + self.img_dy * y_pgain, self.g_pitch_limit[1]),self.g_pitch_limit[0])
            self.g_yaw = max(min(self.g_yaw + self.img_dx * x_pgain, self.g_yaw_limit[1]), self.g_yaw_limit[0])
            # print(self.simGetCameraInfo("0"))
            self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                   airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))
        else:
            self.img_dx = self.img_human_center[0] - camera_center[0]
            self.img_dy = self.img_human_center[1] - camera_center[1]
            y_pgain = -0.00015
            x_pgain = 0.00015
            self.g_pitch = max(min(self.g_pitch + self.img_dy * y_pgain, self.g_pitch_limit[1]), self.g_pitch_limit[0])
            self.g_yaw = max(min(self.g_yaw + self.img_dx * x_pgain, self.g_yaw_limit[1]), self.g_yaw_limit[0])
            # print(self.simGetCameraInfo("0"))
            self.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                                   airsim.to_quaternion(self.g_pitch, 0, self.g_yaw)))



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
    return pitch_y, roll_x, yaw_z  # in radians

