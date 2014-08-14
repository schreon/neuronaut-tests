'''
Created on Jul 6, 2014

@author: leon
'''



import logging
import unittest

import neuro
from neuro.backpropagation import Backpropagation
from neuro.rmsprop import RMSProp
from neuro.stopping import EarlyStopping
from neuro.training import FullBatchTrainer, SGDTrainer
from neuro.weightdecay import Renormalize
from neuro.dropout import Dropout

from test_base import AbaloneBaseTest

logging.basicConfig(level=logging.INFO)
class Test(AbaloneBaseTest):    
    def testRMSPropFullBatch(self):                
        MyTrainer = neuro.create("RMSProp.FullBatch", 
                                 FullBatchTrainer, 
                                 Backpropagation, 
                                 EarlyStopping, 
                                 RMSProp, 
                                 Renormalize, 
                                 Dropout)
        self.trainer = MyTrainer(min_steps=1000, network=self.net)

#     def testRMSPropSGD(self):                
#         MyTrainer = neuro.create("RMSProp.SGD", SGDTrainer, Backpropagation, EarlyStopping, RMSProp, Renormalize)
#         self.trainer = MyTrainer(min_steps=1000, network=self.net, minibatch_size=100)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCore']
    unittest.main()
