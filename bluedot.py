import numpy
import matplotlib.pyplot as plt
from string import Template

import pyopencl as cl
from pyopencl.cltypes import make_double3

mf = cl.mem_flags

from planets import earth

config = {
    'size_x': 1000,
    'size_y': 1000,

    'ray_samples':   16,
    'light_samples': 8,

    'exposure': 2.0,
    'zoom':     2.5,

    'eye_pos': numpy.array([0, -3, 1.1]),
    'eye_dir': numpy.array([0, 1, -0.35])
}

sun_range = (0, 360, 10)

cl_platform = cl.get_platforms()[0]
cl_context = cl.Context(properties=[(cl.context_properties.PLATFORM, cl_platform)])
cl_queue   = cl.CommandQueue(cl_context)

cl_picture = cl.Buffer(cl_context, mf.WRITE_ONLY, size=config['size_x']*config['size_y']*3*numpy.float64(0).nbytes)
program = None

with open('raymarch.cl') as f:
    program = cl.Program(cl_context, Template(f.read()).substitute(
        {**config, **earth}
    )).build()

for i in numpy.arange(*sun_range):
    sun = make_double3(numpy.cos(i*2*numpy.pi/360),numpy.sin(i*2*numpy.pi/360),0)
    print(sun)

    program.render(
        cl_queue, (config['size_x'], config['size_y']), None, cl_picture,
        make_double3(*(config['eye_pos'] * earth['earth_radius'])),
        make_double3(*(config['eye_dir'] * earth['earth_radius'])),
        sun)

    np_picture = numpy.ndarray(shape=(config['size_y'], config['size_x'], 3), dtype=numpy.float64)
    cl.enqueue_copy(cl_queue, np_picture, cl_picture).wait();

    plt.imsave("bluedot_%05.1f.png" % i, np_picture, origin='lower')
