import logging
import unittest
from neuro.rprop import RPROP
from neuro.softmax import SoftmaxLayer
from neuro.weightdecay import Renormalize
import numpy

from neuro.backpropagation import Backpropagation
from neuro.classification import Classification, classification_delta_kernel, class_errors, ClassificationTrainer

from neuro.model import FeedForwardNeuralNetwork, LogisticLayer, NaNMask
import neuro
from neuro.cuda import CUDAContext
from neuro.training import SGDTrainer, FullBatchTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.model import DenseLayer
import mnist
import os


logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testClassificationKernel(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        out = numpy.zeros((5,10), dtype=numpy.float32)
        err = numpy.zeros((5,10), dtype=numpy.int32)
        tar = numpy.zeros((5,), dtype=numpy.int32)
        tar[0] = 1
        error = numpy.zeros((1,), dtype=numpy.int32)
        out, tar, err, error = ctx.upload(out, tar, err, error)

        class_errors(ctx, tar, out, err)
        ctx.sum(err, error)

        logging.info(error.get())


    def testClassification(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        inp, targ, inpt, targt = ctx.upload(*mnist.get_patterns())
        assert inp.dtype == inpt.dtype == numpy.float32
        assert targ.dtype == targt.dtype == numpy.int32

        NetworkClass = neuro.create("MyNetwork", FeedForwardNeuralNetwork, Classification)

        net = NetworkClass(context=ctx, input_shape=(784,))

        LogisticDense = neuro.create("MyDenseLayer", DenseLayer, LogisticLayer)
        SoftmaxDense = neuro.create("MySoftMaxLayer", DenseLayer, SoftmaxLayer)

        net.add_layer(LayerClass=LogisticDense, num_units=512)
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