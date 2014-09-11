import numpy
from prettyplotlib import brewer2mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import logging

from PIL import Image

green_purple = brewer2mpl.get_map('PRGn', 'diverging', 11).mpl_colormap
colornorm = Normalize(vmin=-6, vmax=6)
colormap = ScalarMappable(norm=colornorm, cmap=green_purple).to_rgba

def array2colors(array):
    return numpy.uint8(colormap(array)*255)

def colors2image(colors, scale):
    colors = numpy.repeat(numpy.repeat(colors, scale, axis=1), scale, axis=0)
    return Image.fromarray(colors)

def weights_to_square(w):
    w = w.ravel()
    k = len(w)
    m = int(numpy.ceil(numpy.sqrt(k)))
    nw = numpy.empty((m**2,))
    nw[:] = numpy.nan
    nw[:k] = w
    return nw.reshape(m,m)

class WeightsPlotter(object):
    def __init__(self, scale=10):
        self.scale = scale
        self.canvas = None

    def plot_weights(self, weights):
        weights = [numpy.concatenate((w, b.reshape(1, -1)), axis=0) for (w, b) in weights]

        if self.canvas is None:
            height = max([w.shape[0] for w in weights])
            width = numpy.sum([w.shape[1] for w in weights]) + 5*len(weights)
            self.canvas = numpy.empty((height, width, 4), dtype=numpy.uint8)
            self.canvas[:] = numpy.nan

        off_x = 0
        for i, w in enumerate(weights):
            width = w.shape[1]
            height = w.shape[0]
            self.canvas[:height,off_x:off_x+width] = array2colors(w)
            off_x += width + 5

        return self.canvas



###

import matplotlib.animation as animation
from pylab import *
from multiprocessing import Process, Queue

class IterableQueue(object):
    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        while True:
            x = self.queue.get()
            if x is None:
                break
            else:
                yield x

    def __len__(self):
        return 999999999999999 # workaround ...


def render_animation(file_name, queue):
    plotter = WeightsPlotter()
    fig = plt.figure(facecolor='gray')
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = {}
    def update_img(array):
        array = plotter.plot_weights(array)
        if im.get('im', None) is None:
            im['im'] = ax.imshow(array,cmap=green_purple,interpolation='nearest', vmin=-6.0, vmax=6.0)
            im['im'].set_clim([0, 1])
            fig.set_size_inches([19.20, 10.80])
            tight_layout()
        im['im'].set_data(array)
        return im['im']

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img, frames=IterableQueue(queue))
    writer = animation.writers['ffmpeg'](fps=30, bitrate=27*1024)

    ani.save(file_name, writer=writer, extra_args=['-vcodec', 'libx264'])

class AnimationRender(Process):
        def __init__(self, file_name):
            self.queue = Queue(maxsize=100)
            super(AnimationRender, self).__init__(target=render_animation, args=(file_name, self.queue))

        def add_frame(self, frame):
            self.queue.put(frame)

        def join(self):
            self.queue.put(None) # signal the end
            return super(AnimationRender, self).join()
