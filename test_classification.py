import logging
import unittest

from neuro.history import History
from neuro.softmax import Softmax
import numpy
from neuro.backpropagation import BackpropagationTrainer
from neuro.classification import ClassificationNetwork, class_errors
from neuro.model import FeedForwardNeuralNetwork
from neuro.logistic import Logistic
import neuro
from neuro.cuda import CUDAContext
from neuro.training import SGDTrainer
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.dense import DenseLayer
from plot_weights import AnimationRender
import mnist


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

        class LogisticDenseLayer(Logistic, DenseLayer):
            pass

        class SoftmaxDenseLayer(Softmax, DenseLayer):
            pass

        net.add_layer(LogisticDenseLayer, num_units=512)
        net.add_layer(LogisticDenseLayer, num_units=256)
        net.add_layer(SoftmaxDenseLayer, num_units=10)

        ar = AnimationRender("data/mnist.webm")
        ar.start()

        class MyTrainer(RMSProp, EarlyStopping, BackpropagationTrainer, History, SGDTrainer):
            def train_step(self, *args, **kwargs):
                super(MyTrainer, self).train_step(*args, **kwargs)
                if self.steps % 10 == 0: # only every 10th frame gets drawn
                    ar.add_frame(self.network.download())

        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt))
        net.reset()
        trainer.train()
        ar.join()

