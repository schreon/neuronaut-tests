import logging
import unittest
import time
from scipy import signal
import numpy

from neuro.backpropagation import Backpropagation
from neuro.classification import Classification, ClassificationTrainer

from neuro.convolution import Convolution2DLayer, convolve, get_output_shape
from neuro.maxpool import Maxpool2DLayer

from neuro.model import FeedForwardNeuralNetwork, LogisticLayer
import neuro
from neuro.cuda import CUDAContext
from neuro.softmax import SoftmaxLayer
from neuro.training import SGDTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.model import DenseLayer
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

        NetworkClass = neuro.create("MyNetwork", FeedForwardNeuralNetwork, Classification)

        net = NetworkClass(context=ctx, input_shape=(28,28))

        ConvLayer = neuro.create("MyConvLayer", Convolution2DLayer, LogisticLayer)
        MPLayer = neuro.create("MyMaxpool", Maxpool2DLayer)
        LogisticDense = neuro.create("MyDenseLayer", DenseLayer, LogisticLayer)
        SoftmaxDense = neuro.create("MySoftMaxLayer", DenseLayer, SoftmaxLayer)

        net.add_layer(LayerClass=ConvLayer, num_units=16)
        net.add_layer(LayerClass=MPLayer)
        net.add_layer(LayerClass=ConvLayer, num_units=16)
        net.add_layer(LayerClass=MPLayer)
        net.add_layer(LayerClass=ConvLayer, num_units=16)
        net.add_layer(LayerClass=LogisticDense, num_units=256)
        net.add_layer(LayerClass=SoftmaxDense, num_units=10)


        TrainerClass = neuro.create("MyTrainer",
                                 SGDTrainer,
                                 ClassificationTrainer,
                                 Backpropagation,
                                 EarlyStopping,
                                 RMSProp
                                 )

        trainer = TrainerClass(network=net)

        state = trainer.TrainingState(network=net, size=128)
        test_state = trainer.TestState(network=net, size=inpt.shape[0])

        net.reset_weights()
        trainer.train(state, test_state, inp, targ, inpt, targt)

        # TODO: implement convolution
        # TODO: implement classification output
        # TODO: implement maxpooling


    def test_convolution_core(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        # number of images
        n = 128

        # input dimensions
        height_channels = 28
        width_channels = 29
        number_channels = 1

        # output dimensions
        number_filters= 32
        height_filters = 4
        width_filters = 5

        a_ref = numpy.random.randn(n, number_channels, height_channels, width_channels).astype(numpy.float32)
        w_ref = numpy.random.randn(number_channels, number_filters, height_filters, width_filters).astype(numpy.float32)

        for mode in ['full', 'valid', 'same']:
            logging.info("Mode: %s" % mode)
            b_ref = numpy.zeros(get_output_shape(a_ref, w_ref, mode)).astype(numpy.float32)

            num_steps = 2

            _, _, _, o_width, o_height = get_output_shape(a_ref, w_ref, mode)
            num_ops = o_width*o_height*height_filters*width_filters*n*number_channels*number_filters


            start_time = time.time()
            for _ in xrange(num_steps):
                for i in range(n):
                    for c in range(number_channels):
                        for f in range(number_filters):
                            b_ref[i, c, f] = signal.convolve2d(a_ref[i, c], w_ref[c, f], mode=mode)
            current_time = time.time()

            gflops = 1.0e-9 * num_ops * num_steps / (current_time - start_time)
            steps_per_sec = n*num_steps / (current_time - start_time)
            logging.info("CPU: " + mode + " convolutions / second: %.4f, msec / convoltion: %.4f, gflops: %.4f" % (steps_per_sec,1000.0 / steps_per_sec, gflops))


            a, w, b = ctx.upload(a_ref, w_ref, numpy.empty_like(b_ref))

            ctx.synchronize()
            # warmup
            for _ in range(5):
                convolve(ctx, a, w, b, mode=mode)
            ctx.synchronize()

            num_steps = 1000
            ctx.synchronize()
            start_time = time.time()
            for _ in xrange(num_steps):
                convolve(ctx, a, w, b, mode=mode)
            ctx.synchronize()
            current_time = time.time()

            gflops = 1.0e-9 * num_ops * num_steps / (current_time - start_time)
            steps_per_sec = n*num_steps / (current_time - start_time)
            logging.info("GPU: " + mode + " convolutions / second: %.4f, msec / convoltion: %.4f, gflops: %.4f" % (steps_per_sec,1000.0 / steps_per_sec, gflops))

            bravel_ref = b_ref.sum(axis=0).ravel()
            bravel = b.get().sum(axis=0).ravel()
            assert len(bravel_ref) == len(bravel)
            for i in range(len(bravel)):
                if numpy.abs(bravel_ref[i] - bravel[i]) > 0.0001:
                    raise(Exception("mismatch at %d, expected: %.5f, actual: %.5f" % (i, bravel_ref[i], bravel[i])))

            logging.info(mode + " passed: no mismatch")

    def test_connect(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        # number of images
        n = 2

        # input dimensions
        height_channels = 7
        width_channels = 8
        number_channels = 4

        # output dimensions
        number_filters = 5
        height_filters = 3
        width_filters = 4

        a_ref = numpy.random.randn(n, number_channels, height_channels, width_channels).astype(numpy.float32)
        w_ref = numpy.random.randn(number_channels, number_filters, height_filters, width_filters).astype(numpy.float32)

        for mode in ['valid', 'full', 'same']:
            logging.info("Mode: %s" % mode)
            b_ref = numpy.zeros(get_output_shape(a_ref, w_ref, mode)).astype(numpy.float32)
            c_ref = numpy.sum(b_ref, axis=0)

            #### GPU ####
            num_steps = 1
            a, w, b, c = ctx.upload(a_ref, w_ref, numpy.empty_like(b_ref), numpy.empty_like(c_ref))

            ctx.synchronize()
            # warmup
            for _ in range(5):
                convolve(ctx, a, w, b, mode=mode)
                ctx.sum(b, c, axis=0)
            ctx.synchronize()
            start_time = time.time()
            for _ in xrange(num_steps):
                convolve(ctx, a, w, b, mode=mode)
                ctx.sum(b, c, axis=0)
            ctx.synchronize()
            current_time = time.time()
            steps_per_sec = n*num_steps / (current_time - start_time)
            logging.info("GPU: " + mode + " propagations / second: %.4f, msec / propagations: %.4f" % (steps_per_sec,1000.0 / steps_per_sec))

            ##### CPU ####
            num_steps = 1
            c_ref = None
            start_time = time.time()
            for _ in xrange(num_steps):
                for i in range(n):
                    for ch in range(number_channels):
                        for f in range(number_filters):
                            b_ref[i, ch, f] = signal.convolve2d(a_ref[i, ch], w_ref[ch, f], mode=mode)
                # sum over channels
                c_ref = numpy.sum(b_ref, axis=0)
            current_time = time.time()
            steps_per_sec = n*num_steps / (current_time - start_time)
            logging.info("CPU: " + mode + " propagations / second: %.4f, msec / propagations: %.4f" % (steps_per_sec,1000.0 / steps_per_sec))


            cravel_ref = c_ref.ravel()
            cravel = c.get().ravel()
            assert len(cravel_ref) == len(cravel)
            for i in range(len(cravel)):
                if numpy.abs(cravel_ref[i] - cravel[i]) > 0.01:
                    raise(Exception("mismatch at %d, expected: %.5f, actual: %.5f" % (i, cravel_ref[i], cravel[i])))

            logging.info(mode + " passed: no mismatch")

    def test_backprop_memory_layout(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        # number of images
        n = 64

        # dimensions
        number_channels = 64
        number_filters = 64
        height_filters = 3
        width_filters = 3

        height_channels = 8
        width_channels = 8

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
        logging.info(deltas.shape)
        deltas_intermediate = numpy.zeros(get_output_shape(deltas, weights, 'backprop')).astype(numpy.float32)
        logging.info(deltas_intermediate.shape)
        prev_deltas = deltas_intermediate.sum(axis=2)
        assert prev_deltas.shape == inputs.shape
        logging.info(prev_deltas.shape)
        gradient_intermediate = numpy.zeros(get_output_shape(prev_deltas, deltas, 'gradient'))
        logging.info(gradient_intermediate.shape)
        gradient = gradient_intermediate.sum(axis=0)
        logging.info(gradient.shape)
        assert gradient.shape == weights.shape
        logging.info("---------------")


        arrays = (inputs, weights, activations_intermediate, activations, deltas, deltas_intermediate, prev_deltas, gradient_intermediate, gradient)
        inputs, weights, activations_intermediate, activations, deltas, deltas_intermediate, prev_deltas, gradient_intermediate, gradient = ctx.upload(*arrays)

        # TODO: in the propagation case, weights are shared. in the delta case not!
        # convolve deltas over inputs, mode, resulting in prev_deltas