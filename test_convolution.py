import logging
import unittest
import time
from neuro.classification import ClassificationNetwork
from neuro.dense import DenseLayer
from neuro.history import History
from neuro.logistic import LogisticLayer
from scipy import signal
import numpy

from neuro.backpropagation import BackpropagationTrainer

from neuro.convolution import Convolution2DLayer, get_output_shape, convolve2d_propagation, convolve2d_backprop, \
    convolve2d_gradient
from neuro.maxpool import Maxpool2DLayer

from neuro.model import FeedForwardNeuralNetwork
import neuro
from neuro.cuda import CUDAContext
from neuro.softmax import SoftmaxLayer
from neuro.training import SGDTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
import mnist
from numpy.testing import assert_almost_equal
import os


logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testConvolution(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        inp, targ, inpt, targt = ctx.upload(*mnist.get_patterns())
        assert inp.dtype == inpt.dtype == numpy.float32
        assert targ.dtype == targt.dtype == numpy.int32

        NetworkClass = neuro.create("MyNetwork", FeedForwardNeuralNetwork, ClassificationNetwork)

        net = NetworkClass(context=ctx, input_shape=(1,28,28))

        ConvLayer = neuro.create("MyConvLayer", Convolution2DLayer, LogisticLayer)
        MPLayer = neuro.create("MyMaxpool", Maxpool2DLayer)
        LogisticDense = neuro.create("MyDenseLayer", DenseLayer, LogisticLayer)
        SoftmaxDense = neuro.create("MySoftMaxLayer", DenseLayer, SoftmaxLayer)

        net.add_layer(ConvLayer, num_units=16)
        net.add_layer(MPLayer)
        net.add_layer(ConvLayer, num_units=16)
        net.add_layer(MPLayer)
        net.add_layer(ConvLayer, num_units=16)
        net.add_layer(LogisticDense, num_units=256)
        net.add_layer(SoftmaxDense, num_units=10)


        class MyTrainer(RMSProp, EarlyStopping, BackpropagationTrainer, History, SGDTrainer):
            pass

        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=None)
        net.reset()
        trainer.train()


    def test_convolution_memory_layout(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        # number of images
        n = 32

        # dimensions
        number_channels = 16
        height_channels = 14
        width_channels = 14
        number_filters = 32
        height_filters = 3
        width_filters = 3

        # propagation
        inputs = numpy.random.randn(n, number_channels, height_channels, width_channels).astype(numpy.float32)
        weights = numpy.random.randn(number_channels, number_filters, height_filters, width_filters).astype(numpy.float32)
        activations_intermediate = numpy.zeros(get_output_shape(inputs, weights, 'propagation')).astype(numpy.float32)
        activations = numpy.sum(activations_intermediate, axis=1)

        logging.info("Propagation:")
        logging.info(inputs.shape)
        logging.info(weights.shape)
        logging.info(activations_intermediate.shape)
        logging.info(activations.shape)
        logging.info("----------------")

        # backprop
        logging.info("Backpropagation:")

        deltas = numpy.empty_like(activations)
        deltas_intermediate = numpy.zeros(get_output_shape(deltas, weights, 'backprop')).astype(numpy.float32)
        prev_deltas = deltas_intermediate.sum(axis=2)
        gradient_intermediate = numpy.zeros(get_output_shape(prev_deltas, deltas, 'gradient'))
        gradient = gradient_intermediate.sum(axis=0).sum(axis=0).sum(axis=0)

        logging.info(deltas.shape)
        logging.info(deltas_intermediate.shape)
        logging.info(prev_deltas.shape)
        logging.info(gradient_intermediate.shape)
        logging.info(gradient.shape)
        assert prev_deltas.shape == inputs.shape
        assert gradient.shape == weights.shape
        logging.info("---------------")

        arrays = (inputs, weights, activations_intermediate, activations, deltas, deltas_intermediate, prev_deltas, gradient_intermediate, gradient)
        inputs, weights, activations_intermediate, activations, deltas, deltas_intermediate, prev_deltas, gradient_intermediate, gradient = ctx.upload(*arrays)

        def step():
            convolve2d_propagation(ctx, inputs, weights, activations_intermediate)
            ctx.sum(activations_intermediate, activations, axis=1)
            convolve2d_backprop(ctx, deltas, weights, deltas_intermediate)
            ctx.sum(deltas_intermediate, prev_deltas, axis=2)
            convolve2d_gradient(ctx, prev_deltas, deltas, gradient_intermediate)
            # sum over images and the delta field
            ctx.sum(gradient_intermediate, gradient, axis=(0,1,2))

        num_steps = 100
        # warmup
        ctx.synchronize()
        for _ in range(5):
            step()
        ctx.synchronize()
        start_time = time.time()
        for _ in xrange(num_steps):
            step()
        ctx.synchronize()
        current_time = time.time()
        steps_per_sec = n*num_steps / (current_time - start_time)
        logging.info("GPU: backprops / second: %.4f, msec / backprops: %.4f" % (steps_per_sec,1000.0 / steps_per_sec))