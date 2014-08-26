import logging
import unittest
from neuro.history import History
from neuro.rprop import RPROP
from neuro.softmax import SoftmaxLayer
from neuro.weightdecay import Renormalize
import numpy

from neuro.backpropagation import BackpropagationTrainer
from neuro.classification import ClassificationNetwork, classification_delta_kernel, class_errors

from neuro.model import FeedForwardNeuralNetwork, NaNMask
from neuro.logistic import LogisticLayer
import neuro
from neuro.cuda import CUDAContext
from neuro.training import SGDTrainer, FullBatchTrainer, Trainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.dense import DenseLayer
import mnist
import os


logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testClassificationKernel(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        out = numpy.zeros((5,10), dtype=numpy.float32)
        err = numpy.zeros((5,), dtype=numpy.int32)
        tar = numpy.zeros((5,), dtype=numpy.int32)
        tar[0] = 1
        error = numpy.zeros((1,), dtype=numpy.int32)
        out, tar, err, error = ctx.upload(out, tar, err, error)

        class_errors(ctx, tar, out, err)
        ctx.sum(err, error)

        logging.info(error.get())


    def testClassification(self):
        ctx = CUDAContext()

        inp, targ, inpt, targt = ctx.upload(*mnist.get_patterns())
        assert inp.dtype == inpt.dtype == numpy.float32
        assert targ.dtype == targt.dtype == numpy.int32

        class MyNetwork(ClassificationNetwork, FeedForwardNeuralNetwork):
            pass

        net = MyNetwork(context=ctx, input_shape=(784,))

        class LogisticDenseLayer(LogisticLayer, DenseLayer):
            pass

        class SoftmaxDenseLayer(SoftmaxLayer, DenseLayer):
            pass

        net.add_layer(LogisticDenseLayer, num_units=512)
        net.add_layer(LogisticDenseLayer, num_units=256)
        net.add_layer(SoftmaxDenseLayer, num_units=10)

        class MyTrainer(RMSProp, EarlyStopping, BackpropagationTrainer, History, SGDTrainer):
            pass

        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt))
        net.reset()
        trainer.train()