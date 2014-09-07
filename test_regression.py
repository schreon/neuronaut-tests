import logging
import unittest

from neuro.backpropagation import BackpropagationTrainer
from neuro.dense import DenseLayer
from neuro.history import History
from neuro.logistic import LogisticLayer

from neuro.model import FeedForwardNeuralNetwork, RegressionNetwork
import neuro
from neuro.cuda import CUDAContext
from neuro.rprop import RPROP
from neuro.training import SGDTrainer, FullBatchTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
import abalone
import numpy
import os


logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testRegression(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        inp, targ, inpt, targt = ctx.upload(*abalone.get_patterns())
        logging.info(inp.shape)
        logging.info(targ.shape)

        assert inp.dtype == inpt.dtype == numpy.float32
        assert targ.dtype == targt.dtype == numpy.float32

        class MyNetwork(RegressionNetwork, FeedForwardNeuralNetwork):
            pass

        net = MyNetwork(context=ctx, input_shape=(8,))

        class LogisticDenseLayer(LogisticLayer, DenseLayer):
            pass


        net.add_layer(LogisticDenseLayer, num_units=512)
        net.add_layer(LogisticDenseLayer, num_units=256)
        net.add_layer(LogisticDenseLayer, num_units=1)

        class MyTrainer(RPROP, EarlyStopping, BackpropagationTrainer, History, FullBatchTrainer):
            pass

        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt))
        net.reset()
        trainer.train()