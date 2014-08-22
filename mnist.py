'''
Created on Jun 4, 2014

@author: schreon
'''
import numpy
import os

import os
import struct
from array import array
import logging
log = logging.getLogger("MNIST")

class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                         os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                         os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in xrange(size):
            images.append([0] * rows * cols)

        for i in xrange(size):
            images[i][:] = image_data[i * rows * cols : (i + 1) * rows * cols]

        return images, labels

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        log.info('Showing num:%d' % train_label[0])
        log.info(self.display(train_img[0]))
        return True

    @classmethod
    def display(cls, img, width=28):
        render = ''
        for i in range(len(img)):
            if i % width == 0: render += '\n'
            if img[i] > 200:
                render += '1'
            else:
                render += '0'
        return render

def _download_file(url, file_name):
    import urllib2
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    log.info("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        log.info(status)

    f.close()


def _load(path, file_name):
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        fn_gz = file_path + '.gz'
        directory = os.path.dirname(fn_gz)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(fn_gz):
            _download_file("http://yann.lecun.com/exdb/mnist/" + file_name + '.gz', fn_gz)

        import gzip
        # unpack
        inF = gzip.open(fn_gz, 'rb')
        outF = open(file_path, 'wb')
        outF.write(inF.read())
        inF.close()
        outF.close()

def _shuffle_in_unison(mat1, mat2):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(mat1)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(mat2)

def get_patterns():
    import cPickle
    mnist_pik = os.path.join('data', 'mnist.pik')
    if not os.path.exists(mnist_pik):
        log.info("creating pickle file")
        # make sure the mnist files are present
        _load(os.path.join('data'), 'train-images-idx3-ubyte')
        _load(os.path.join('data'), 'train-labels-idx1-ubyte')
        _load(os.path.join('data'), 't10k-images-idx3-ubyte')
        _load(os.path.join('data'), 't10k-labels-idx1-ubyte')

        mndata = MNIST(os.path.join('data'))
        inputs, targets = mndata.load_training()
        inputs_test, targets_test = mndata.load_testing()

        # normalize by maximum
        inputs = numpy.array(inputs).astype(numpy.float32) / 255.0
        inputs_test =  numpy.array(inputs_test).astype(numpy.float32) / 255.0

        targets = numpy.array(targets, dtype=numpy.int32).ravel()
        targets_test = numpy.array(targets_test, dtype=numpy.int32).ravel()

        # shuffle inputs heavily
        _shuffle_in_unison(inputs, targets)
        _shuffle_in_unison(inputs, targets)
        _shuffle_in_unison(inputs, targets)
        _shuffle_in_unison(inputs, targets)
        _shuffle_in_unison(inputs, targets)

        cPickle.dump((inputs, targets, inputs_test, targets_test), open(mnist_pik, "wb"), protocol=-1)
    else:
        log.info("loading from pickle file")
        inputs, targets, inputs_test, targets_test = cPickle.load(open(mnist_pik, "rb"))

    return inputs, targets.astype(numpy.int32), inputs_test, targets_test.astype(numpy.int32)

if __name__ == "__main__":
    log.info("mnist standalone download")
    inputs, targets, inputs_test, targets_test = get_patterns()
