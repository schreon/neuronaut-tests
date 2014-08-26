'''
Created on Aug 19, 2014

@author: schreon
'''
import unittest
import logging
import mnist
import neuro
from neuro.model import FeedForwardNeuralNetwork
from neuro.cuda import CUDAContext
from neuro.model import DenseLayer, LogisticLayer
from neuro.convolution import Convolution2DLayer
from neuro.maxpool import Maxpool2DLayer
from neuro.training import SGDTrainer
from neuro.backpropagation import BackpropagationTrainer
from neuro.stopping import EarlyStopping
from neuro.rmsprop import RMSProp

logging.basicConfig(level=logging.INFO)

class Test(unittest.TestCase):


    def testMnist(self):
        ctx = neuro.create("MyContext", CUDAContext)()
        inputs, targets, inputs_test, targets_test = mnist.get_patterns()
        
        inputs = inputs.reshape(-1, 28, 28)
        inputs_test = inputs_test.reshape(-1, 28, 28)
        
        inputs, targets = ctx.upload(inputs, targets)
        
        
        NetworkClass = neuro.create("ConvoNetwork", FeedForwardNeuralNetwork)
        
        LogisticConvolutionLayer = neuro.create("LogConvLayer", Convolution2DLayer, LogisticLayer)
        LogisticDenseLayer = neuro.create("LogDenseLayer", DenseLayer, LogisticLayer)
        
        net = NetworkClass(context=ctx, input_shape=(28,28))
        net.add_layer(LayerClass=LogisticConvolutionLayer, filter_shape=(3,3))
        net.add_layer(LayerClass=Maxpool2DLayer, filter_shape=(2,2))
        net.add_layer(LayerClass=LogisticConvolutionLayer, filter_shape=(3,3))
        net.add_layer(LayerClass=Maxpool2DLayer, filter_shape=(2,2))
        net.add_layer(LayerClass=LogisticConvolutionLayer, filter_shape=(3,3))
        net.add_layer(LayerClass=Maxpool2DLayer, filter_shape=(2,2))
        net.add_layer(LayerClass=LogisticDenseLayer, num_units=256)
        
        TrainerClass = neuro.create("MyTrainer", 
                         SGDTrainer, 
                         BackpropagationTrainer,
                         EarlyStopping, 
                         RMSProp
                         )
        trainer = TrainerClass(network=net)
        training_state = trainer.TrainingState(network=net, size=16)
        logging.info(net.shape)
        for act in training_state.activations:
            if act is not None:
                logging.info(act.shape)
        
        logging.info("-----")
        act = training_state.activations[-2]
        
        logging.info(act.shape)
        act_reshaped = neuro.reshape(act, (16,9))
        act_reshaped2 = neuro.reshape(act, (16,9))
        assert(id(act_reshaped) == id(act_reshaped2))
        logging.info("-----")
        
        logging.info("weights:")
        for layer in net.layers:
            logging.info(layer.weights.shape)
        
        trainer.train(training_state, None, inputs, targets, inputs_test, targets_test)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMnist']
    unittest.main()