from haversine import haversine
from math import sin, pi, cos, asin, sqrt, atan

point_origin = (37.1214124, 127.112)
x_D = point_origin[0]
y_D = point_origin[1]

dx, dy = 5, 10

earth_radius = 6378137.0  # Radius of "spherical" earth

# Coordinate offsets in radians
dLat = dx / earth_radius * 180 / pi
dLon = dy / (earth_radius * cos(pi * x_D / 180)) * 180 / pi

# 조정값
x_H = x_D + dLat
y_H = y_D + dLon
haversine_dist = haversine((x_D, y_D), (x_H, y_H), unit='m')
print('Drone : ', (x_D, y_D))
print('Human : ', (x_H, y_H))
print('Distance : ', (dx, dy), sqrt(dx ** 2 + dy ** 2))
print('Haversine Dist : ', haversine_dist)
print('error : ', (haversine_dist - sqrt(dx ** 2 + dy ** 2)))

