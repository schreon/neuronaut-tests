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
from neuro.weightdecay import Renormalize
import numpy
from plot_weights import WeightsPlotter, AnimationRender
import json

logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testRegression(self):
        ctx = CUDAContext()

        inp, targ, inpt, targt, minimum, maximum = abalone.get_patterns()
        inp, targ, inpt, targt = ctx.upload(inp, targ, inpt, targt)

        logging.info(inp.shape)
        logging.info(targ.shape)

        assert inp.dtype == inpt.dtype == numpy.float32
        assert targ.dtype == targt.dtype == numpy.float32

        class MyNetwork(RegressionNetwork, FeedForwardNeuralNetwork):
            pass

        net = MyNetwork(context=ctx, input_shape=(8,))

        class LogisticDenseLayer(Logistic, DenseLayer):
            pass

        net.add_layer(LogisticDenseLayer, num_units=8)
        net.add_layer(LogisticDenseLayer, num_units=4)
        net.add_layer(LogisticDenseLayer, num_units=1)

        ar = AnimationRender("data/abalone.webm")
        ar.start()

        class MyTrainer(Renormalize, RPROPMinus, BackpropagationTrainer, History, FullBatchTrainer):
            def on_new_best(self, old_best, new_best):
                super(MyTrainer, self).on_new_best(old_best, new_best)


            def train_step(self, *args, **kwargs):
                super(MyTrainer, self).train_step(*args, **kwargs)
                if self.steps % 10 == 0: # only every 10th frame gets drawn
                    ar.add_frame(self.network.download())


        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt))
        trainer.parameters['l2_decay'] = 0.0001
        net.reset()
        trainer.train()
        ar.join()

        with open('data/abalone.json', 'wb') as fp:
            hist = trainer.errors['history']['test']
            hist = zip(range(1,len(hist)*trainer.validation_frequency+1, trainer.validation_frequency), hist)
            json.dump(hist, fp)