# coding=utf-8
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
import time
import os
from mako.template import Template

logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testRegression(self):
        ctx = CUDAContext()

        dir_name = os.path.join("reports", "report-abalone-"+ time.strftime("%Y%m%d-%H%M%S"))
        vid_name = "training.webm"
        os.mkdir(dir_name)

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

        net.add_layer(LogisticDenseLayer, num_units=32)
        net.add_layer(LogisticDenseLayer, num_units=16)
        net.add_layer(LogisticDenseLayer, num_units=1)

        ar = AnimationRender(os.path.join(dir_name, vid_name))
        ar.start()

        class MyTrainer(Renormalize, EarlyStopping, History, RPROPMinus, BackpropagationTrainer, FullBatchTrainer):
            def on_new_best(self, old_best, new_best):
                super(MyTrainer, self).on_new_best(old_best, new_best)


            def train_step(self, *args, **kwargs):
                super(MyTrainer, self).train_step(*args, **kwargs)
                if self.steps % 10 == 0: # only every 10th frame gets drawn
                    ar.add_frame(self.network.download())

        trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt), l2_day=0.0001)
        net.reset()
        trainer.train()
        ar.join()

        hist = trainer.errors['history']['test']
        hist = zip(range(1,len(hist)*trainer.validation_frequency+1, trainer.validation_frequency), hist)

        logging.info(trainer.parameters)
        data = dict(history=hist, filename="abalone", time=time.asctime(), parameters=trainer.parameters)
        data_string = json.dumps(data)
        html = Template(filename="neuronaut-report.html", input_encoding='utf-8').render(data=data_string)
        with open(os.path.join(dir_name, "report.html"), "wb") as f:
            f.write(html.encode('utf-8'))
