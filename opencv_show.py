# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path 
import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import time
import sys

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")

cameraType = "scene"

for arg in sys.argv[1:]:
  cameraType = arg.lower()

cameraTypeMap = { 
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}

if (not cameraType in cameraTypeMap):
  printUsage()
  sys.exit(0)

print (cameraTypeMap[cameraType])

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# client.takeoffAsync().join()

help = False

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime=time.clock()

fps = 0

while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    for i in range(1):
        rawImage = client.simGetImage(str(i), cameraTypeMap[cameraType])
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            # png_copy1 = png.copy()
            # png_copy1[:,:,0] = 0
            # png_copy2 = png.copy()
            # png_copy2[:, :, 1] = 0
            # png_copy3 = png.copy()
            # png_copy3[:, :, 2] = 0
            # png_copy4 = png.copy()
            # png_copy4[:, :, 3] = 0

            cv2.putText(png,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
            # cv2.imshow('0', png_copy1)
            # cv2.imshow('1', png_copy2)
            # cv2.imshow('2', png_copy3)
            # cv2.imshow('3', png_copy4)
            cv2.imshow(str(i), png[:,:,:3])


    frameCount  = frameCount  + 1
    endTime=time.clock()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime
    
    key = cv2.waitKey(1) & 0xFF;
    if (key == 27 or key == ord('q') or key == ord('x')):
        break;
