import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import folium
import webbrowser


def plot_results(dir):
    df = pd.read_csv(dir / "simulation_data.csv")
    plt.figure(1)
    est_p_x = df['estimated_personcoord_record'].apply(lambda x: eval(x)[0])
    est_p_y = df['estimated_personcoord_record'].apply(lambda x: eval(x)[1])
    act_p_x = df['actual_personcoord_record'].apply(lambda x: eval(x)[0])
    act_p_y = df['actual_personcoord_record'].apply(lambda x: eval(x)[1])
    plt.scatter(est_p_y, est_p_x, label='estimated_coord', s=1)
    plt.scatter(act_p_y, act_p_x, label='actual_coord', s=1)
    plt.axes().set_aspect(1)
    plt.legend()
    plt.grid()
    plt.title("actual & eatimated person record")
    plt.savefig(dir / 'actual & eatimated person.png')

    plt.figure(2)
    gps_lon = df['human_lon_lat_record'].apply(lambda x: eval(x)[0])
    gps_lat = df['human_lon_lat_record'].apply(lambda x: eval(x)[1])
    gps_lon_real = df['human_lon_lat_real_record'].apply(lambda x: eval(x)[0])
    gps_lat_real = df['human_lon_lat_real_record'].apply(lambda x: eval(x)[1])
    plt.scatter(gps_lon, gps_lat, label='human_est', s=1)
    plt.scatter(gps_lon_real, gps_lat_real, label='human_real', s=1)
    plt.title("human_lon_lat_record")
    plt.legend()
    plt.grid()
    plt.axes().set_aspect(1/math.cos(gps_lat_real[0]))
    plt.savefig(dir / 'human_lon_lat_record.png')

    plt.figure(3)
    est_err_dist = df['error_distance_est_record']
    real_err_dist = df['error_distance_real_record']
    plt.plot(est_err_dist, label='est_err_dist')
    plt.plot(real_err_dist, label='real_err_dist')
    plt.legend()
    plt.grid()
    plt.title("error distance")
    plt.savefig(dir / 'error distance.png')

    plt.figure(4)
    target_x = df['drone_target_record'].apply(lambda x: eval(x)[0])
    target_y = df['drone_target_record'].apply(lambda x: eval(x)[1])
    plt.scatter(target_y, target_x, label='human_est', s=1)
    plt.title("drone_target_record")
    plt.legend()
    plt.grid()
    plt.savefig(dir / 'drone_target_record.png')

    m = folium.Map(
        location=[gps_lat[0], gps_lon[0]],
        zoom_start=100,
        max_zoom=23
    )

    for i, latlon in enumerate(zip(gps_lat, gps_lon)):
        lat, lon = latlon
        folium.CircleMarker(
            location=(lat, lon),
            radius=0.1,
            fill=True,  # Set fill to True
            fill_color='#000000',
            color='#000000',
            popup=f"HUMAN_est:{i} lat:{lat:.6f} lon:{lon:.6f}"
        ).add_to(m)

    for i, latlon in enumerate(zip(gps_lat_real, gps_lon_real)):
        lat, lon = latlon
        folium.CircleMarker(
            location=(lat, lon),
            radius=0.1,
            fill=True,  # Set fill to True
            fill_color='#00FF00',
            color='#00FF00',
            popup=f"HUMAN_real:{i} lat:{lat:.6f} lon:{lon:.6f}"
        ).add_to(m)

    m.save(str(dir / "map.html"))
    webbrowser.open(dir / "map.html")
    plt.show()
