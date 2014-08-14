'''
Created on Aug 14, 2014

@author: schreon
'''
import unittest


class Test(unittest.TestCase):


    def testEvaluator(self):
        ctx = neuro.create("MyContext", CUDAContext)(num_workers=4)    
        inp, targ, minimum, maximum = abalone.get_patterns()

        # test/train ratio 50% : 50%
        n = int(0.5 * inp.shape[0])
        inp_test = inp[:n]
        targ_test = targ[:n]        
        inp = inp[n:]
        targ = targ[n:]
        
        data_train = (inp, targ)
        data_test = (inp_test, targ_test)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
