import numpy
from string import Template

import pyopencl as cl
from pyopencl.cltypes import make_double3

mf = cl.mem_flags

class Renderer:
    def __init__(self, config, planet):
        self.config = config
        self.planet = planet

        self.cl_platform = cl.get_platforms()[0]
        self.cl_context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.cl_platform)])
        self.cl_queue   = cl.CommandQueue(self.cl_context)

        self.cl_picture = cl.Buffer(self.cl_context, mf.WRITE_ONLY, size=self.config['size_x']*self.config['size_y']*3*numpy.float64(0).nbytes)

        with open('firmament/raymarch.cl') as f:
            self.program = cl.Program(self.cl_context, Template(f.read()).substitute(
                {**self.config, **self.planet}
            )).build()

    def render_pinhole(self, sun):
        self.program.render_pinhole(
            self.cl_queue,
            (self.config['size_x'], self.config['size_y']),
            None,
            self.cl_picture,
            make_double3(*(self.config['eye_pos'] * self.planet['earth_radius'])),
            make_double3(*(self.config['eye_dir'] * self.planet['earth_radius'])),
            make_double3(*sun))

        np_picture = numpy.ndarray(shape=(self.config['size_y'], self.config['size_x'], 3), dtype=numpy.float64)
        cl.enqueue_copy(self.cl_queue, np_picture, self.cl_picture).wait();
        return np_picture

    def render_fisheye(self, sun):
        self.program.render_fisheye(
            self.cl_queue,
            (self.config['size_x'], self.config['size_y']),
            None,
            self.cl_picture,
            make_double3(*(self.config['eye_pos'] * self.planet['earth_radius'])),
            make_double3(*(self.config['eye_dir'] * self.planet['earth_radius'])),
            make_double3(*sun))

        np_picture = numpy.ndarray(shape=(self.config['size_y'], self.config['size_x'], 3), dtype=numpy.float64)
        cl.enqueue_copy(self.cl_queue, np_picture, self.cl_picture).wait();
        return np_picture
