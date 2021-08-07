import matplotlib.pyplot as plt
import numpy as np
import math

# TODO
#  1. 방향키 누르면 순서 앞뒤로 옮겨가면서 인덱스 따라서 plot되게 하는건 어떨까
#  2. 로그파일을 하나 만들어서 파일 따로만드는게 아니라 전부다 하나의 파일에 들어가도록 만들어야함

plot_for1 = "actual_personcoord_record"
plot_for2 = "estimated_personcoord_record"
# have to change exp number every execution
save_dir = "C:\\seungwoo\\KUAV\\KUAV_airsim\\yolo_drone\\runs\\detect\\exp21\\"


if plot_for1 == "error_distance" or plot_for2 == "error_distance":
    f = open(save_dir+"error_distance.txt", 'r')
    record = []
    plt.figure(1)
    while True:
        line = f.readline()
        if not line: break
        line = eval(line)
        # print(line)
        record.append(line)

    f.close()
    plt.plot(record)
    plt.title("error_distance")

if plot_for1 == "error_distance_real" or plot_for2 == "error_distance_real":
    f = open(save_dir+"error_distance_real.txt", 'r')
    record = []
    plt.figure(1)
    while True:
        line = f.readline()
        if not line: break
        line = eval(line)
        # print(line)
        record.append(line)

    f.close()
    plt.plot(record)
    plt.title("error_distance_real")

if plot_for1 == "trace" or plot_for2 == "trace":
    f = open(save_dir+"recordings/trace.txt", 'r')
    Hrecord = []
    Drecord = []
    plt.figure(1)
    while True:
        line = f.readline()
        if not line : break
        line = eval(line)
        drone, human = line[0], line[1]
        # print(line)
        Hrecord.append(human)
        Drecord.append(drone)
    f.close()
    Hrecord = np.array(Hrecord)
    Drecord = np.array(Drecord)

    Hrecord.reshape((-1,2))
    Drecord.reshape((-1,2))

    plt.scatter(Hrecord[:,0],Hrecord[:,1])
    plt.scatter(Drecord[:,0],Drecord[:,1])

if plot_for1 == "actual_personcoord_record" or plot_for2 == "actual_personcoord_record":
    f = open(save_dir+"actual_personcoord_record.txt", 'r')
    record = []
    plt.figure(1)
    while True:
        line = f.readline()
        if not line: break
        line = eval(line)
        line = [line[0],line[1]]
        # print(line)
        record.append(line)
    record = np.array(record)
    f.close()
    plt.scatter(record[:,0],record[:,1],label = 'actural_coord',s = 1)
    plt.legend()

if plot_for1 == "estimated_personcoord_record" or plot_for2 == "estimated_personcoord_record":
    f = open(save_dir+"estimated_personcoord_record.txt", 'r')
    record1 = []
    plt.figure(1)
    while True:
        line = f.readline()
        if not line: break
        line = eval(line)
        line = [line[0],line[1]]
        # print(line)
        record1.append(line)
    record1 = np.array(record1)
    f.close()
    plt.scatter(record1[:,0],record1[:,1],label = 'estimated_coord',s = 1)
    plt.scatter(0, 0, label='Drone Pose', s=5)
    plt.legend()

if plot_for1 == "drone_target_record" or plot_for2 == "drone_target_record":
    f = open(save_dir+"drone_target_record.txt", 'r')
    record2 = []
    plt.figure(1)
    while True:
        line = f.readline()
        if not line: break
        line = eval(line)
        line = [line[0],line[1]]
        # print(line)
        record2.append(line)
    record2 = np.array(record2)
    f.close()
    # plt.scatter(record2[:,0],record2[:,1],label = 'drone_target_record',s = 1)
    # plt.scatter(0, 0, label='Drone Pose', s=5)
    plt.legend()

plt.show()

#
# f = open("recordings/estimated_personcoord_record.txt", 'r')
# record = []
# plt.figure(1)
# while True:
#     line = f.readline()
#     if not line: break
#     line = eval(line)
#     line = [line[0],line[1]]
#     # print(line)
#     record.append(line)
#     record = np.array(record)
#     f.close()
#     plt.scatter(record[:,0],record[:,1],label = 'estimated_coord',s = 1)
#     plt.scatter(0, 0, label='Drone Pose', s=5)
#     plt.legend()
