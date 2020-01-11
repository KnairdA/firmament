import numpy as np
import matplotlib.pyplot as plt
from string import Template

import pyopencl as cl
from pyopencl.cltypes import make_double3

mf = cl.mem_flags

from planets import earth

from sun import sun_direction
from datetime import datetime

config = {
    'size_x': 1000,
    'size_y': 1000,

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

cl_platform = cl.get_platforms()[0]
cl_context = cl.Context(properties=[(cl.context_properties.PLATFORM, cl_platform)])
cl_queue   = cl.CommandQueue(cl_context)

cl_picture = cl.Buffer(cl_context, mf.WRITE_ONLY, size=config['size_x']*config['size_y']*3*np.float64(0).nbytes)
program = None

with open('raymarch.cl') as f:
    program = cl.Program(cl_context, Template(f.read()).substitute(
        {**config, **earth}
    )).build()

for time in np.arange(*time_range):
    pit = datetime(*config['date'], int(np.floor(time)), int((time-np.floor(time))*60), 0)
    sun_dir = sun_direction(config['latitude'], config['longitude'], pit, config['timezone'], 1.0 if config['summertime'] else 0.0)

    sun = make_double3(
        np.cos(sun_dir[0])*np.sin(sun_dir[1]),
        np.cos(sun_dir[0])*np.cos(sun_dir[1]),
        np.sin(sun_dir[0])
    )
    print(sun_dir)

    program.render_fisheye(
        cl_queue, (config['size_x'], config['size_y']), None, cl_picture,
        make_double3(*(config['eye_pos'] * earth['earth_radius'])),
        make_double3(*(config['eye_dir'] * earth['earth_radius'])),
        sun)

    np_picture = np.ndarray(shape=(config['size_y'], config['size_x'], 3), dtype=np.float64)
    cl.enqueue_copy(cl_queue, np_picture, cl_picture).wait();

    fig = plt.gcf()

    ax_image = fig.add_axes([0.0, 0.0, 1.0, 1.0], label='Sky')
    ax_image.imshow(np_picture, origin='lower')
    ax_image.axis('off')

    ax_polar = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='polar', label='Overlay')
    ax_polar.patch.set_alpha(0)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rlim(bottom=90, top=0)
    yticks = [0, 15, 30, 45, 60, 75, 90]
    ax_polar.set_yticks(yticks)
    ax_polar.set_yticklabels(['' if i == 90 else '%d°' % i for i in yticks], color='white', fontsize=6)
    ax_polar.set_rlabel_position(90/4)
    ax_polar.set_xticklabels(['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'])
    ax_polar.grid(True)

    plt.savefig("sky_%05.1f.png" % time, bbox_inches='tight', pad_inches=0.2)

    fig.clear()

