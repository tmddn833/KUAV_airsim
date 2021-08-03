# KUAV
19회 로봇항공기 경연대회

- human tracing drone simulation with yolov5
- repository for C:\AirSim\PythonClient\multirotor
- yolov5 clone of https://github.com/ultralytics/yolov5
- main simulation python file is project_crop_yolo.py

## project_crop_yolo.py
Execute the python file after running the simulation with airsimgamemode.  
How to launch airsimgamemode is to set the worldsetting - gamemode - game overwriting to airsim game mode in unreal engine.  
Details in https://microsoft.github.io/AirSim/apis/
- execute command
> python project_crop_yolo.py --source 0

## drone_control.py MyMultirotorClient class
In this simulation, this class is mainly used. 
Main functions that perform the mission are as follows.
- mission_start : In arbitrary starting place, drone move to the closest point to human with given gps coordinate.
- adjust_target_gps : find the human coord from the image detection, and adjust target coord drone have to move to.  
- adjust_gimbal_angle : to fix the human image to center of the camera image, camera(gimbal angle) corrected in every loop with detected pixel information.  
