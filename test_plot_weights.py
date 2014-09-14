import logging
import unittest

from neuro.backpropagation import BackpropagationTrainer
from neuro.dense import DenseLayer
from neuro.history import History
from neuro.logistic import Logistic

from neuro.model import FeedForwardNeuralNetwork, RegressionNetwork
from neuro.cuda import CUDAContext
from neuro.rprop import RPROPMinus
from neuro.training import FullBatchTrainer
from neuro.stopping import EarlyStopping
import abalone
import numpy
from neuronaut_plot import plot_weight_matrix

logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testPlotWeights(self):
        ctx = CUDAContext()

        inp, targ, inpt, targt = ctx.upload(*abalone.get_patterns())
        logging.info(inp.shape)
        logging.info(targ.shape)

        assert inp.dtype == inpt.dtype == numpy.float32
        assert targ.dtype == targt.dtype == numpy.float32

        class MyNetwork(RegressionNetwork, FeedForwardNeuralNetwork):
            pass

        net = MyNetwork(context=ctx, input_shape=(8,))

        class LogisticDenseLayer(Logistic, DenseLayer):
            pass

        net.add_layer(LogisticDenseLayer, num_units=512)
        net.add_layer(LogisticDenseLayer, num_units=256)
        net.add_layer(LogisticDenseLayer, num_units=1)

        class MyTrainer(RPROPMinus, EarlyStopping, BackpropagationTrainer, History, FullBatchTrainer):
            pass

        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt))
        net.reset()

        np_weights = net.download()

        for (w,b) in np_weights:
            plot_weight_matrix(w, b)