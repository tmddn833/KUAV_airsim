from haversine import haversine
from math import sin, pi, cos, asin, sqrt, atan
import folium
from IPython.display import Image, display
import io
from PIL import Image
import time
import matplotlib.pyplot as plt
import matplotlib.image as image

point_origin = (37.587004, 127.031719)


dx, dy = 1, 1

earth_radius = 6378137.0  # Radius of "spherical" earth

# Coordinate offsets in radians
dLat = dx / earth_radius * 180 / pi
dLon = dy / (earth_radius * cos(pi * point_origin[0] / 180)) * 180 / pi

# 조정값
x_H = point_origin[0] + dLat
y_H = point_origin[1] + dLon
haversine_dist = haversine((point_origin[0], point_origin[1]), (x_H, y_H), unit='m')
print('Drone : ', (point_origin[0], point_origin[1]))
print('Human : ', (x_H, y_H))
print('Distance : ', (dx, dy), sqrt(dx ** 2 + dy ** 2))
print('Haversine Dist : ', haversine_dist)
print('error : ', (haversine_dist - sqrt(dx ** 2 + dy ** 2)))

m = folium.Map((x_H, y_H), zoom_start=100)


from selenium import webdriver
import os
from io import BytesIO
import cv2
import numpy as np

fn='testmap.html'
tmpurl='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=fn)
m.save(fn)

# print(type(png))
# png = m._to_png(0.1)
# png=Image.open(BytesIO(png))
# png.show()

browser = webdriver.Chrome("C:\seungwoo\chromedriver.exe")
browser.get(tmpurl)
#Give the map tiles some time to load
time.sleep(5)
marker = folium.CircleMarker(
    location=[x_H, y_H],
    radius=10)
marker.add_to(m)
m.save(fn)
time.sleep(5)
marker = folium.CircleMarker(
    location=[x_H+1, y_H+1],
    radius=10)
marker.add_to(m)
m.save(fn)
#
# png = browser.get_screenshot_as_png()
#
#
#
# png = np.frombuffer(png, dtype = np.uint8)
# png = cv2.imdecode(png, cv2.IMREAD_COLOR)
#
#
# print(png.shape)
# cv2.imshow('test', png)
# cv2.waitKey(0)
