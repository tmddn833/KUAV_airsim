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
from detect_crop import parse_opt
from drone_control import MyMultirotorClient


def printUsage():
    print("Usage: python camera.py [depth|segmentation|scene]")


cameraType = "scene"

cameraTypeMap = {
    "depth": airsim.ImageType.DepthVis,
    "segmentation": airsim.ImageType.Segmentation,
    "seg": airsim.ImageType.Segmentation,
    "scene": airsim.ImageType.Scene,
    "disparity": airsim.ImageType.DisparityNormalized,
    "normals": airsim.ImageType.SurfaceNormals
}
print(cameraTypeMap[cameraType])


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, client=client, cameraType=cameraType)
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
    crop_detect = False  # if the model do detect in crop image
    total_detect = False  # if the model do detect in original image
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

        # make crop version
        if not total_detect or n == 0:  # In this condition, the crop detection occur
            img0_crop = im0s.copy()
            img0_crop = img0_crop[0]
            cropsize = 50
            # image shape y,x (height and width)
            lowerbound_x = max(client.img_human_center[0] - cropsize, 0)
            higherbound_x = min(client.img_human_center[0] + cropsize, img0_crop.shape[1])
            lowerbound_y = max(client.img_human_center[1] - cropsize, 0)
            higherbound_y = min(client.img_human_center[1] + cropsize, img0_crop.shape[0])

            img0_crop = img0_crop[lowerbound_y:higherbound_y, lowerbound_x:higherbound_x]
            img0_crop = [img0_crop]
            img_crop = [letterbox(x, imgsz, stride=stride)[0] for x in img0_crop]
            img_crop = np.stack(img_crop, 0)
            img_crop = img_crop[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img_crop = np.ascontiguousarray(img_crop)

            # print(img.shape) # (1, 3, 480, 640)
            img_crop = torch.from_numpy(img_crop).to(device)
            img_crop = img_crop.half() if half else img_crop.float()  # uint8 to fp16/32
            img_crop /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_crop.ndimension() == 3:
                img_crop = img_crop.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred_raw = model(img, augment=augment)[0]
        if not total_detect or n == 0:  # crop condition
            pred_raw_crop = model(img_crop, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred_raw, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        if not total_detect or n == 0:  # crop condition
            pred_crop = non_max_suppression(pred_raw_crop, conf_thres / 3, iou_thres, classes, agnostic_nms,
                                            max_det=max_det)

        t2 = time_synchronized()

        # Process detections
        for i, dets in enumerate(zip(pred, pred_crop)):  # iteration per source(one loop for one source)
            det, det_crop = dets
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                if not total_detect or n == 0:  # crop condition
                    im0_crop = img0_crop[i].copy()
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                if not total_detect or n == 0:  # crop condition
                    det_crop[:, :4] = scale_coords(img_crop.shape[2:], det_crop[:, :4], im0_crop.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results for crop
                client.human_detect = False
                crop_detect = False
                if not total_detect or n == 0:  # crop condition
                    for *xyxy, conf, cls in reversed(det_crop):
                        if view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            center, foot = plot_one_box2(xyxy, im0_crop, label=label, color=colors(c, True),
                                                         line_thickness=1)
                            if c == 0:
                                # print(center, lowerbound_x, lowerbound_y, im0.shape, im0_crop.shape)
                                # (132, 106) 540 380 (960, 1280, 3) (200, 200, 3) -> the ratio always same, just add
                                center = (lowerbound_x + center[0], lowerbound_y + center[1])
                                foot = (lowerbound_x + foot[0], lowerbound_y + foot[1])
                                client.img_human_center = center
                                client.img_human_foot = foot
                                client.human_detect = True
                                crop_detect = True
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Write results
                total_detect = False
                if crop_detect is False:
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            center, foot = plot_one_box2(xyxy, im0, label=label, color=colors(c, True),
                                                         line_thickness=1)
                            if c == 0:
                                client.img_human_center = center
                                client.img_human_foot = foot
                                client.human_detect = True
                                total_detect = True
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 2
            textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)

            if view_img:
                if crop_detect:
                    # cv2.imshow("Crop", im0_crop)
                    # qim0[im0.shape[0]-im0_crop.shape[0]:im0.shape[0],0:im0_crop.shape[1], :] = im0_crop
                    im0[lowerbound_y:higherbound_y, lowerbound_x:higherbound_x] = im0_crop
                    cv2.putText(im0, 'Crop', (lowerbound_x, lowerbound_y + textSize[1]), fontFace,
                                fontScale,
                                (255, 0, 255), thickness)
                # print((human_center[0], im0.shape[0] / 2), (im0.shape[1] / 2, im0.shape[0] / 2))
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
                cv2.putText(im0, 'Crop Detection ' + str(crop_detect), (10, 25 + textSize[1]), fontFace, fontScale,
                            (255, 0, 255), thickness)
                cv2.putText(im0, ' Total Detection ' + str(total_detect), (10, 40 + textSize[1]), fontFace, fontScale,
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

            # adjust_target_gps()
            # adjust_gimbal_angle(client, human_center=human_center, camera_center=(im0.shape[1] / 2, im0.shape[0] / 2))

        # try:
        #     print(client.img_human_center)
        # except:
        #     pass
        n += 1  # count up the iteration

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    # here is initial settings for simulation
    client = MyMultirotorClient(default_gimbal_pitch=-math.pi / 4,  # how much drone will look down at spawn
                                xdFoV=63 / 180 * math.pi,
                                hovering_altitude=-10,  # meter
                                velocity_gain=0.3,  #
                                track_target=True,  #
                                plot_threading = True # plot the trajectory
                                )
    client.armDisarm(True)
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    # Go to initial location
    # client.takeoffAsync(timeout_sec='5').join()
    client.moveToPositionAsync(client.start.x_val, client.start.y_val, client.hovering_altitude, 5).join()
    target = client.simGetObjectPose('NPC_3')
    client.mission_start((target.position.x_val, target.position.y_val), coordinate='XYZ')

    # Trace the target human
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt), client=client)
    os.startfile(str(client.save_dir))



