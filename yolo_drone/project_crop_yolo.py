# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode


# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python

import time
import math
import sys
import os
import argparse
from pathlib import Path

import cv2
import airsim
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box, plot_one_box2
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox
from drone_control import MyMultirotorClient
from gpsmap.airsim_gpsflask import GpsFlask


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=320,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        client=None  # if it is airsim, use client for source
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    client.save_dir = save_dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, client=client)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # to check the detection
    client.human_detect = False  # if the model do detect
    n = 0  # the number of detect iteration
    client.img_human_center = (int(dataset.imgs[0].shape[1] / 2), int(dataset.imgs[0].shape[0] / 2))
    client.img_human_foot = client.img_human_center
    for path, img, im0s, vid_cap in dataset:
        # print(img.shape) # (1, 3, 480, 640)
        # print(im0s[0].shape) # (480, 640, 3)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference & Apply NMS
        t1 = time_synchronized()
        pred_raw = model(img, augment=augment)[0]
        pred = non_max_suppression(pred_raw, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()
        client.human_detect = False
        # Process detections
        for i, det in enumerate(pred):  # iteration per source(one loop for one source)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results for crop
                min_len = im0.shape[1]**2+im0.shape[0]**2
                temp_h_detec = False
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        center, foot = plot_one_box2(xyxy, im0, label=label, color=colors(c, True),
                                                     line_thickness=1)
                        if c == 0:
                            foot_len = math.sqrt((foot[0]-im0.shape[1]/2)**2 + (foot[1]-im0.shape[0]/2)**2)
                            if min_len > foot_len:
                                min_len = foot_len
                                client.img_human_center = center
                                client.img_human_foot = foot
                                temp_h_detec = True
                if temp_h_detec:
                    client.read_sim_info()
                    client.img_dx = client.img_human_foot[0] - client.w / 2
                    client.img_dy = client.img_human_foot[1] - client.h / 2
                    client.human_detect = True
                                # total_detect = True
                # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s), {client.img_human_foot}')

            # Stream results
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 2
            textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)

            if view_img:
                # if total_detect:
                #     cv2.rectangle(im0, [lowerbound_x, lowerbound_y], [higherbound_x, higherbound_y],
                #                   (0, 255, 0), -1, cv2.LINE_4)
                cv2.line(im0,
                         (int(client.img_dx + im0.shape[1] / 2), int(im0.shape[0] / 2)),
                         (int(im0.shape[1] / 2), int(im0.shape[0] / 2)), (0, 0, 255),
                         thickness=1, lineType=cv2.LINE_AA)
                cv2.line(im0,
                         (int(im0.shape[1] / 2), int(client.img_dy + im0.shape[0] / 2)),
                         (int(im0.shape[1] / 2), int(im0.shape[0] / 2)), (0, 0, 255),
                         thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(im0, 'Human Detection ' + str(client.human_detect), (10, 10 + textSize[1]), fontFace,
                            fontScale,
                            (255, 0, 255), thickness)
                cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond
                key = cv2.waitKey(1) & 0xFF
                if (key == 27 or key == ord('q') or key == ord('x')):
                    break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        n += 1  # count up the iteration

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default="C:\seungwoo\KUAV\yolov5\\runs\\train\exp25\weights\\best.pt", help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default="C:\seungwoo\KUAV\yolov5\\visdrone_trained_model\weights\\best.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(exclude=('tensorboard', 'thop'))
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    client = MyMultirotorClient(default_gimbal_pitch=-math.pi / 4,  # how much drone will look down at spawn
                                xdFoV=63 / 180 * math.pi,
                                hovering_altitude=-30,  # meter
                                velocity_gain=0.5,  #
                                track_target=True,  #
                                plot_traj=True  # plot the trajectory
                                )

    # Go to initial location
    target = client.simGetObjectPose('NPC_3')
    # Move to starting position
    # client.mission_start((target.position.x_val, target.position.y_val),(target.position.x_val+20, target.position.y_val),
    #                      coordinate='XYZ')
    client.mission_start((37.587363326113646, 127.03149227523414),(37.58754298917047, 127.03149227523414), coordinate='GPS')

    # Gps posting flask start
    GpsFlask(client)
    # Run the yolo model, including tracing the human if track_target = True
    # With 'q' keyboard input, the 'run' will finish.
    run(**vars(opt), client=client)
    # If the mission finish, Open the data directory and plot the results.
    os.startfile(str(client.save_dir))
    client.dataplot()
