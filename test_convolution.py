import logging
import unittest

from neuro.backpropagation import Backpropagation

from neuro.convolution import Convolution2DLayer
from neuro.maxpool import Maxpool2DLayer

from neuro.model import FeedForwardNeuralNetwork, LogisticLayer
import neuro
from neuro.cuda import CUDAContext
from neuro.training import SGDTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.model import DenseLayer
import mnist


logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testConvolution(self):
        ctx = neuro.create("MyContext", CUDAContext)()

        inp, targ, inpt, targt = ctx.upload(*mnist.get_patterns())
        NetworkClass = neuro.create("MyNetwork", FeedForwardNeuralNetwork)

        net = NetworkClass(context=ctx, input_shape=(28,28))

        ConvLayer = neuro.create("MyConvLayer", Convolution2DLayer, LogisticLayer)
        MPLayer = neuro.create("MyMaxpool", Maxpool2DLayer)
        LogisticDense = neuro.create("MyDenseLayer", DenseLayer, LogisticLayer)
        net.add_layer(LayerClass=ConvLayer, num_units=16)
        net.add_layer(LayerClass=MPLayer)
        net.add_layer(LayerClass=ConvLayer, num_units=16)
        net.add_layer(LayerClass=MPLayer)
        net.add_layer(LayerClass=ConvLayer, num_units=16)
        net.add_layer(LayerClass=LogisticDense, num_units=1)

        TrainerClass = neuro.create("MyTrainer",
                                 SGDTrainer,
                                 Backpropagation,
                                 EarlyStopping,
                                 RMSProp
                                 )

        trainer = TrainerClass(network=net)

        state = trainer.TrainingState(network=net, size=32)
        trainer.train(state, None, inp, targ, None, None)

        # TODO: implement convolution
        # TODO: implement classification output
        # TODO: implement maxpooling
