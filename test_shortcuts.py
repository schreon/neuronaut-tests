'''
Created on Aug 15, 2014

@author: schreon
'''
import unittest
import logging
import abalone
import neuro
from neuro.cuda import CUDAContext
from neuro.shortcuts import RegressionNetworkFactory


logging.basicConfig(level=logging.INFO)


class Test(unittest.TestCase):

    def testName(self):
        ctx = neuro.create("MyContext", CUDAContext)()
        data = abalone.get_patterns()
        factory = RegressionNetworkFactory(ctx)
        factory.add_layer(256)
        factory.add_layer(256)
        Cls, weights = factory.train(data, validation_ratio=0.5)        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()