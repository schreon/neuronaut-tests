import logging
import unittest
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

        # input dimensions
        height_channels = 5
        width_channels = 5
        number_channels = 2

        # output dimensions
        number_filters=4
        height_filters = 3
        width_filters = 3

        # full convolution
        a_ref = numpy.random.randn(number_channels, height_channels, width_channels).astype(numpy.float32)
        w_ref = numpy.random.randn(number_channels, number_filters, height_filters, width_filters).astype(numpy.float32)

        for mode in ['full', 'valid', 'same']:
            logging.info("Mode: %s" % mode)
            b_ref = numpy.zeros(get_output_shape(a_ref, w_ref, mode)).astype(numpy.float32)

            logging.info(b_ref.shape)

            for c in range(number_channels):
                for f in range(number_filters):
                    b_ref[c,f] = signal.convolve2d(a_ref[c], w_ref[c,f], mode=mode)


            logging.info(b_ref.shape)

            a, w, b = ctx.upload(a_ref, w_ref, numpy.empty_like(b_ref))

            convolve(ctx, a, w, b, mode=mode)

            bravel_ref = b_ref.ravel()
            bravel = b.get().ravel()
            assert len(bravel_ref) == len(bravel)
            logging.info("%d elements" % len(bravel))
            wrong = 0
            for i in range(len(bravel)):
                if numpy.abs(bravel_ref[i] - bravel[i]) > 0.00001:
                    logging.info("mismatch at %d, expected: %.5f, actual: %.5f" % (i, bravel_ref[i], bravel[i]))
                    wrong += 1

            logging.info("%d of %d elements are wrong" % (wrong, len(bravel)))