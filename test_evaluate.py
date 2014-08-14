'''
Created on Jul 23, 2014

@author: schreon
'''
import logging
import abalone
import unittest

import numpy

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
from neuro import OpenClContext


logging.basicConfig(level=logging.INFO)


class Test(unittest.TestCase):


    def testEvaluate(self):
        ctx = neuro.create("MyContext", CUDAContext)(num_workers=4)    
        inp, targ, minimum, maximum = abalone.get_patterns()
        
        #inp = numpy.random.uniform(0.0, 1.0, (50*1024, 1024)).astype(numpy.float32)
        #targ = numpy.random.uniform(0.0, 1.0, (50*1024, 48)).astype(numpy.float32)
        
        # test/train ratio 50% : 50%
        n = int(0.5*inp.shape[0])
        inp_test = inp[:n]
        targ_test = targ[:n]        
        inp = inp[n:]
        targ = targ[n:]
        
        data_train = (inp, targ)
        data_test = (inp_test, targ_test)
        
        MyNetwork = neuro.create("MyNetwork",
                                 FeedForwardNeuralNetwork,
                                 Regression,
                                 NaNMask,
                                 )
        network_args = dict(context=ctx, input_shape=inp.shape[1])
                 
        MyTrainer = neuro.create("MyTrainer", 
                                 FullBatchTrainer, 
                                 Backpropagation, 
                                 EarlyStopping, 
                                 RPROP, 
                                 Renormalize
                                 )
        trainer_args = dict(min_steps=1000,
                            validation_frequency=5, # validate the model every 5 steps
                            validate_train=True, # also validate on training set
                            logging_frequency=2.0, # log progress every 2 seconds
                            max_weight_size=15.0,
                            max_step_size=0.01
                            )
        
        network_structure = [
           dict(num_units=64, function=ctx.softplus, derivative=ctx.softplus_derivative),
           dict(num_units=64, function=ctx.softplus, derivative=ctx.softplus_derivative),
           dict(num_units=targ.shape[1], function=ctx.linear, derivative=ctx.linear_derivative)
        ]
        
        evaluator = ConfigurationEvaluator(
          data_train=data_train,
          data_test=data_test,
          NetworkClass=MyNetwork,
          TrainerClass=MyTrainer,
          network_args=network_args,
          trainer_args=trainer_args,
          network_structure=network_structure
          )

        evaluator.evaluate(num_evaluations=1, file_name="data/Abalone-Test.pik")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testEvaluate']
    unittest.main()