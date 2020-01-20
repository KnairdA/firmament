import numpy
import matplotlib.pyplot as plt

from firmament import Renderer
from firmament.planets import earth

config = {
    'size_x': 1920//4,
    'size_y': 1080//4,

    'ray_samples'  : 32,
    'light_samples': 8,

    'exposure': 2.0,
    'zoom':     1.0,

    'eye_pos': numpy.array([0, 0, 1.0001]),
    'eye_dir': numpy.array([0, 1, 0])
}

sun_range = (-10, 90, 10)

renderer = Renderer(config, earth)

for i in numpy.arange(*sun_range):
    sun = (0.0, numpy.cos(i*2*numpy.pi/360), numpy.sin(i*2*numpy.pi/360))
    print(sun)

    np_picture = renderer.render_pinhole(sun)

    plt.imsave("sky_%05.1f.png" % (i-sun_range[0]), np_picture, origin='lower')
