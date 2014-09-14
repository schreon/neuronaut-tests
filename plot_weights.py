import numpy
from prettyplotlib import brewer2mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import logging

from PIL import Image

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

def render_confusion(file_name, queue, vmin, vmax, divergent, array_shape):
    from pylab import plt
    import matplotlib.animation as animation
    plt.close()
    fig = plt.figure()

    def update_img((expected, output)):
        plt.cla()
        plt.ylim((vmin, vmin+vmax))
        plt.xlim((vmin, vmin+vmax))
        ax = fig.add_subplot(111)
        plt.plot([vmin, vmin+vmax], [vmin, vmin+vmax])
        ax.grid(True)
        plt.xlabel("expected output")
        plt.ylabel("network output")
        plt.legend()

        expected = expected*vmax + vmin
        output = output*vmax + vmin
        #scat.set_offsets((expected, output))
        scat = ax.scatter(expected, output)
        return scat

    ani = animation.FuncAnimation(fig, update_img, frames=IterableQueue(queue))

    ani.save(file_name, fps=30, extra_args=['-vcodec', 'libvpx', '-threads', '4', '-b:v', '1M'])

def render_weights(file_name, queue, vmin, vmax, divergent, array_shape):
    from pylab import plt
    import matplotlib.animation as animation

    plotter = WeightsPlotter()
    fig = plt.figure(facecolor='gray', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')


    if divergent:
        my_colormap = brewer2mpl.get_map('PiYG', 'diverging', 11).mpl_colormap
    else:
        my_colormap = brewer2mpl.get_map('OrRd', 'sequential', 9).mpl_colormap

    my_colormap.set_bad('#6ECFF6', 1.0)

    im = {}
    def update_img(array):
        array = plotter.plot_weights(array)
        if im.get('im', None) is None:
            im['im'] = ax.imshow(array, cmap=my_colormap, interpolation='nearest', vmin=vmin, vmax=vmax)
            aspect = array.shape[0] / float(array.shape[1])
            fig.set_size_inches([7.2, 7.2*aspect]) # 720 pixels wide, variable height
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        im['im'].set_data(array)
        return im['im']

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img, frames=IterableQueue(queue))
    #writer = animation.writers['ffmpeg'](fps=30, bitrate=27*1024)

    ani.save(file_name, fps=30, extra_args=['-vcodec', 'libvpx', '-threads', '4', '-b:v', '1M'])

class AnimationRender(Process):
        def __init__(self, file_name, vmin=-6.0, vmax=6.0, divergent=True, render_function=render_weights, array_shape=()):
            self.queue = Queue(maxsize=100)
            super(AnimationRender, self).__init__(target=render_function, args=(file_name, self.queue, vmin, vmax, divergent, array_shape))

        def add_frame(self, frame):
            self.queue.put(frame)

        def join(self):
            self.queue.put(None) # signal the end
            return super(AnimationRender, self).join()
