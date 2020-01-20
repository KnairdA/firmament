import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from firmament import Renderer
from firmament.planets import earth
from firmament.sun import sun_direction

fish_eye = False

config = {
    'size_x': 1000 if fish_eye else 1920//4,
    'size_y': 1000 if fish_eye else 1080//4,

    'ray_samples'  : 16,
    'light_samples': 8,

    'exposure': 2.0,
    'zoom':     1.0, # only for pinhole view

    'eye_pos': np.array([0, 0, 1.0001]),
    'eye_dir': np.array([0, 1, 0]), # only for pinhole view

    'date': (2020, 1, 20),
    'timezone': 1, # GMT+1
    'summertime': False,

    'latitude': 49.01356,
    'longitude': 8.40444
}

time_range = (6, 20, 0.5)

renderer = Renderer(config, earth)

for time in np.arange(*time_range):
    pit = datetime(*config['date'], int(np.floor(time)), int((time-np.floor(time))*60), 0)
    sun_dir = sun_direction(config['latitude'], config['longitude'], pit, config['timezone'], 1.0 if config['summertime'] else 0.0)

    sun =(
        np.cos(sun_dir[0])*np.sin(sun_dir[1]),
        np.cos(sun_dir[0])*np.cos(sun_dir[1]),
        np.sin(sun_dir[0])
    )
    print(sun_dir)

    np_picture = (renderer.render_fisheye if fish_eye else renderer.render_pinhole)(sun)

    plt.imsave("sky_%05.1f.png" % time, np_picture, origin='lower')
