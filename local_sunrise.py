import numpy as np
import matplotlib.pyplot as plt
from string import Template

import pyopencl as cl
from pyopencl.cltypes import make_double3

mf = cl.mem_flags

from planets import earth

from sun import sun_direction
from datetime import datetime

fish_eye = True

config = {
    'size_x': 1000 if fish_eye else 1920//4,
    'size_y': 1000 if fish_eye else 1080//4,

    'ray_samples'  : 16,
    'light_samples': 8,

    'exposure': 4.0,
    'zoom':     1.0, # only for pinhole view

    'eye_pos': np.array([0, 0, 1.0001]),
    'eye_dir': np.array([0, 1, 0]), # only for pinhole view

    'date': (2020, 1, 20),
    'latitude': 49.01,
    'longitude': 8.4
}

time_range = (5, 20, 1)

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
    sun_dir = sun_direction(config['latitude'], config['longitude'], pit, 1.0)

    sun = make_double3(
        np.cos(sun_dir[0])*np.sin(sun_dir[1]),
        np.cos(sun_dir[0])*np.cos(sun_dir[1]),
        np.sin(sun_dir[0])
    )
    print(sun_dir)

    (program.render_fisheye if fish_eye else program.render_pinhole)(
        cl_queue, (config['size_x'], config['size_y']), None, cl_picture,
        make_double3(*(config['eye_pos'] * earth['earth_radius'])),
        make_double3(*(config['eye_dir'] * earth['earth_radius'])),
        sun)

    np_picture = np.ndarray(shape=(config['size_y'], config['size_x'], 3), dtype=np.float64)
    cl.enqueue_copy(cl_queue, np_picture, cl_picture).wait();

    plt.imsave("sky_%05.1f.png" % time, np_picture, origin='lower')
