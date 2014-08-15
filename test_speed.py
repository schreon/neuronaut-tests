import logging
import time
import unittest

import numpy
from reikna import cluda

from neuro.opencl import OpenClContext
import neuro
from neuro.cuda import CUDAContext
from numpy.testing import assert_almost_equal

logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testSpeed(self):
        ctx = neuro.create("MyContext", OpenClContext)() 
        n = 1024
        m = 48
        k = 51200
        logging.info(m)
        
        a_cpu = numpy.random.randn(n,k).astype(numpy.float32)
        a = ctx.upload(a_cpu)
        b_cpu = numpy.random.randn(k,m).astype(numpy.float32)
        b = ctx.upload(b_cpu)
        c_cpu = numpy.random.randn(n,m).astype(numpy.float32)
        c = ctx.upload(c_cpu)  
        
        d = numpy.random.randn(1,1).astype(numpy.float32)
        d = ctx.upload(d)
        
        ctx.synchronize()
        # warmup
        for _ in range(5):        
            ctx.dot(a, b, c)
        ctx.synchronize()

        num_steps = 200
        ctx.synchronize()    
        start_time = time.time()                
        for _ in xrange(num_steps):
            ctx.dot(a, b, c) 
        ctx.synchronize()      
        current_time = time.time() 
                     
        steps_per_sec = num_steps / (current_time - start_time)
        num_ops = 2 * m*n*k
        gflops = 1.0e-9 * num_ops * num_steps / (current_time - start_time)                                
        logging.info("matrix multiplications / second: %.4f, gflops: %.4f" % (steps_per_sec, gflops))
         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSpeed']
    unittest.main()