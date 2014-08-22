import logging
import unittest

from neuro.backpropagation import Backpropagation

from neuro.model import FeedForwardNeuralNetwork, LogisticLayer, Regression
import neuro
from neuro.cuda import CUDAContext
from neuro.training import SGDTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.model import DenseLayer
import abalone
import os


logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testRegression(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        inp, targ, inpt, targt = ctx.upload(*abalone.get_patterns())
        logging.info(inp.shape)
        logging.info(targ.shape)

        NetworkClass = neuro.create("MyNetwork", FeedForwardNeuralNetwork, Regression)

        net = NetworkClass(context=ctx, input_shape=(8,))

        LogisticDense = neuro.create("MyDenseLayer", DenseLayer, LogisticLayer)

        net.add_layer(LayerClass=LogisticDense, num_units=512)
        net.add_layer(LayerClass=LogisticDense, num_units=256)
        net.add_layer(LayerClass=LogisticDense, num_units=1)

        TrainerClass = neuro.create("MyTrainer",
                                 SGDTrainer,
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
