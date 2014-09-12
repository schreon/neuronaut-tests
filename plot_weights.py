import numpy
from prettyplotlib import brewer2mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import logging

from PIL import Image

my_colormap = brewer2mpl.get_map('PiYG', 'diverging', 11).mpl_colormap
my_colormap.set_bad('#6ECFF6', 1.0)

class WeightsPlotter(object):
    def __init__(self, scale=10):
        self.scale = scale
        self.canvas = None

    def plot_weights(self, weights):
        weights = [numpy.concatenate((w, b.reshape(1, -1)), axis=0) for (w, b) in weights]

        if self.canvas is None:
            height = max([w.shape[0] for w in weights])
            width = numpy.sum([w.shape[1] for w in weights]) + 2*(len(weights)-1)
            self.canvas = numpy.empty((height, width), dtype=numpy.float32)
            self.canvas[:] = numpy.nan

        off_x = 0
        for i, w in enumerate(weights):
            width = w.shape[1]
            height = w.shape[0]
            self.canvas[:height,off_x:off_x+width] = w
            off_x += width + 2

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
    fig = plt.figure(facecolor='gray', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

    im = {}
    def update_img(array):
        array = plotter.plot_weights(array)
        if im.get('im', None) is None:
            im['im'] = ax.imshow(array, cmap=my_colormap, interpolation='nearest', vmin=-6.0, vmax=6.0)
            #im['im'].set_clim([0, 1])
            aspect = array.shape[0] / float(array.shape[1])
            fig.set_size_inches([7.2, 7.2*aspect]) # 720 pixels wide, fit height
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        im['im'].set_data(array)
        return im['im']

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img, frames=IterableQueue(queue))
    #writer = animation.writers['ffmpeg'](fps=30, bitrate=27*1024)

    ani.save(file_name, fps=30, extra_args=['-vcodec', 'libvpx', '-threads', '4', '-b:v', '1M'])

class AnimationRender(Process):
        def __init__(self, file_name):
            self.queue = Queue(maxsize=100)
            super(AnimationRender, self).__init__(target=render_animation, args=(file_name, self.queue))

        def add_frame(self, frame):
            self.queue.put(frame)

        def join(self):
            self.queue.put(None) # signal the end
            return super(AnimationRender, self).join()
