import numpy
import matplotlib.pyplot as plt

from firmament import Renderer
from firmament.planets import earth

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

renderer = Renderer(config, earth)

for i in numpy.arange(*sun_range):
    sun = (numpy.cos(i*2*numpy.pi/360), numpy.sin(i*2*numpy.pi/360), 0)
    print(sun)

    np_picture = renderer.render_pinhole(sun)

    plt.imsave("bluedot_%05.1f.png" % i, np_picture, origin='lower')
