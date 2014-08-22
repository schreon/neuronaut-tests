import logging
import neuro
from neuro.backpropagation import Backpropagation
from neuro.cuda import CUDAContext
from neuro.dropout import Dropout, DropoutNetwork
from neuro.evaluate import ConfigurationEvaluator
from neuro.lwta import LWTANetwork
from neuro.model import FeedForwardNeuralNetwork, Regression, NaNMask
from neuro.rmsprop import RMSProp
from neuro.rprop import RPROP
from neuro.sarprop import SARPROP
from neuro.stopping import EarlyStopping
from neuro.training import FullBatchTrainer, SGDTrainer
from neuro.weightdecay import Renormalize
from neuro.model import DenseLayer, LogisticLayer, LinearLayer
import unittest

import numpy

import abalone
import os


logging.basicConfig(level=logging.INFO)


class Test(unittest.TestCase):


    def testEvaluate(self):
        logging.info(os.environ['PATH'])
        # First, choose whether CUDA or OpenCL should be used as backend.
        # In this example we stick to the CUDAContext. But you can just
        # swap that with OpenCLContext.
        ctx = neuro.create("MyContext", CUDAContext)()
        
        # In the test package, a small regression dataset is prepared.
        # The task is to predict the age of abalone based on 8 attributes.
        inp, targ, minimum, maximum = abalone.get_patterns()

        # You should use a portion of the data for validating
        # the generalization performance of the network.
        # Simply split your numpy arrays along the first dimension:
        n = int(0.5*inp.shape[0])
        inp_test = inp[:n]
        targ_test = targ[:n]        
        inp = inp[n:]
        targ = targ[n:]        
        data_train = (inp, targ)
        data_test = (inp_test, targ_test)
        
        # Now we need to create the network class. neuronaut comes
        # with a factory method which takes a series of Mixin classes
        # and combines them. The first parameter of the create-method
        # is the name you wish for your new class:
        NetworkClass = neuro.create("MyNetwork",
                                 FeedForwardNeuralNetwork,
                                 Regression,
                                 NaNMask,
                                 )
    
        LogisticLayerClass = neuro.create("LogLayer", DenseLayer, LogisticLayer)
        LinearLayerClass = neuro.create("LinLayer", DenseLayer, LinearLayer)
        
        # Next, we must specify the structure of the neural network.
        # Here, we use 2 hidden layers with the logistic activation function
        # and an output layer with linear units.
        network_structure = [
           dict(num_units=inp.shape[1]),
           dict(num_units=64, LayerClass=LogisticLayerClass),
           dict(num_units=64, LayerClass=LogisticLayerClass),
           dict(num_units=targ.shape[1], LayerClass=LinearLayerClass)
        ]

        # To train the network, we also need a Trainer class.
        # In this case we want to do full batch training with
        # the learning algorithm RPROP. To determine when to
        # stop training, we use early stopping. Since the
        # abalone dataset is very small and noisy, we need a
        # regularizer to avoid overfitting. We simply add the
        # Renormalize mixin, which is a special kind of weight
        # decay which constraints the length of the weight vector
        # of each neuron.
        TrainerClass = neuro.create("MyTrainer", 
                                 FullBatchTrainer, 
                                 Backpropagation, 
                                 EarlyStopping, 
                                 RPROP, 
                                 Renormalize
                                 )
        
        
        trainer_options = dict(min_steps=1000,
                            validation_frequency=5, # validate the model every 5 steps
                            validate_train=True, # also validate on training set
                            logging_frequency=2.0, # log progress every 2 seconds
                            max_weight_size=15.0,
                            max_step_size=0.01
                            )

        evaluator = ConfigurationEvaluator(
          data_train=data_train,
          data_test=data_test,
          NetworkClass=NetworkClass,
          TrainerClass=TrainerClass,
          network_args=dict(context=ctx),
          trainer_args=trainer_options,
          network_structure=network_structure
          )

        evaluator.evaluate(num_evaluations=5, file_name="data/Abalone-Test.pik")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testEvaluate']
    unittest.main()