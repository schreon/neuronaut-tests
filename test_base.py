'''
Created on Jul 9, 2014

@author: schreon
'''
from test import abalone
import unittest

from matplotlib import pyplot as plt
import numpy

import neuro
from neuro.cuda import CUDAContext
from neuro.model import FeedForwardNeuralNetwork, Regression, NaNMask
from neuro.dropout import DropoutNetwork


class AbaloneBaseTest(unittest.TestCase):    
    def tearDown(self):        
        inputs, targets, inputs_test, targets_test = self.patterns
        
        trainer = self.trainer
                
        num_patterns_training = inputs.shape[0]
        num_patterns_testing = inputs_test.shape[0]
        
        training_state = trainer.TrainingState(network=self.net, size=num_patterns_training)
        test_state = trainer.TestState(network=self.net, size=num_patterns_testing)

        trainer.train(training_state, test_state, inputs, targets, inputs_test, targets_test)

        plot_args = {
            'name' : self.trainer.__class__.__name__,
            'steps_per_sec' : trainer.steps_per_sec,
            'steps' : trainer.steps,
            'rmse' : trainer.errors['best']['test']
        }
        self.plotResult(test_state.activations[-1].get(), **plot_args)

    def setUp(self):
        ctx = neuro.create("MyContext", CUDAContext)(num_workers=4)    
        inp, targ, minimum, maximum = abalone.get_patterns()
                
        # test/train ratio 50% : 50%
        n = int(0.5*inp.shape[0])
        inp_test = inp[:n]
        targ_test = targ[:n]        
        inp = inp[n:]
        targ = targ[n:]

        self.targets_test = targ_test.ravel()
              
        self.patterns = ctx.upload(inp, targ, inp_test, targ_test)
        self.minimum, self.maximum = minimum, maximum
        
        MyNetwork = neuro.create("MyNetwork", FeedForwardNeuralNetwork, Regression, NaNMask, DropoutNetwork)
        net = MyNetwork(context=ctx, input_shape=inp.shape[1], seed=1234)
 
        net.add_layer(128, ctx.logistic, ctx.logistic_derivative, dropout=0.5)
        net.add_layer(128, ctx.logistic, ctx.logistic_derivative, dropout=0.5)
        net.add_layer(1, ctx.logistic, ctx.logistic_derivative)
        net.reset_weights(std=0.01)
        
        self.net = net
        self.ctx = ctx
    
    def plotResult(self, prediction, **kwargs):
        
        minimum, maximum = self.minimum, self.maximum
        trg = self.targets_test * maximum[-1] + minimum[-1]
        prediction = prediction.ravel() * maximum[-1] + minimum[-1]
        trainer = self.trainer
        
        plt.figure(1, figsize=(15,5))
        plt.subplot(121)
        plt.plot([minimum[-1], minimum[-1]+maximum[-1]], [minimum[-1], minimum[-1]+maximum[-1]], label="ideal", color='g')
        plt.legend()
        plt.scatter(trg, prediction, alpha=0.1, s=50.0)
        plt.grid(True)
        plt.xlabel("target value")
        plt.ylabel("predicted value")
        plt.suptitle("abalone.{name}, test set RMSE: {rmse:.4f}".format(**kwargs))
        plt.title("{steps_per_sec:.2f} steps/sec, {steps} steps until convergence".format(**kwargs))
        plt.subplot(122)
        plt.grid(True)
        plt.plot(range(0, len(trainer.errors['history']['test'])*5, 5), trainer.errors['history']['test'], label="test set")
        plt.plot(range(0, len(trainer.errors['history']['train'])*5, 5), trainer.errors['history']['train'], label="training set")
        plt.legend()  
        plt.title("error progression")
        plt.ylabel("rmse")
        plt.xlabel("#epochs")     
        plt.savefig("data/" + "abalone.{name}.pdf".format(**kwargs))
        plt.close()