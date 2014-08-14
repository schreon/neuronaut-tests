'''
Created on Jul 6, 2014

@author: leon
'''



import logging
import unittest

import neuro
from neuro.backpropagation import Backpropagation
from neuro.dropout import Dropout
from neuro.sarprop import SARPROP
from neuro.stopping import EarlyStopping
from neuro.training import FullBatchTrainer
from neuro.weightdecay import Renormalize
from test_base import AbaloneBaseTest


logging.basicConfig(level=logging.INFO)

class Test(AbaloneBaseTest):    
    def testSARPROPFullBatch(self):                
        MyTrainer = neuro.create("SARPROP.FullBatch", FullBatchTrainer, Backpropagation, EarlyStopping, SARPROP, Renormalize, Dropout)
        self.trainer = MyTrainer(
                                 min_steps=1000, 
                                 network=self.net, 
                                 ini_step_size=10.0, 
                                 max_step_size=10.0, 
                                 l1_decay=0.0,
                                 l2_decay=0.0
                                 )

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCore']
    unittest.main()
