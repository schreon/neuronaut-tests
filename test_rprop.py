'''
Created on Jul 9, 2014

@author: schreon
'''

import logging

import neuro
from neuro.backpropagation import Backpropagation
from neuro.dropout import Dropout
from neuro.rprop import RPROP
from neuro.stopping import EarlyStopping
from neuro.training import FullBatchTrainer
from neuro.weightdecay import Renormalize
from test.test_base import AbaloneBaseTest


logging.basicConfig(level=logging.INFO)
class Test(AbaloneBaseTest):
    
    def testRPROP(self):     
        MyTrainer = neuro.create("RPROP.FullBatch", FullBatchTrainer, Backpropagation, EarlyStopping, RPROP, Renormalize, Dropout)
        self.trainer = MyTrainer(min_steps=1000,
                                 network=self.net,
                                 max_step_size=0.01,
                                 )