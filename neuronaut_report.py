# coding=utf-8
import logging
from neuro.backpropagation import BackpropagationTrainer
from neuro.dense import DenseLayer
from neuro.history import History
from neuro.logistic import Logistic
from neuro.model import FeedForwardNeuralNetwork, RegressionNetwork
from neuro.cuda import CUDAContext
from neuro.rprop import RPROPMinus
from neuro.training import FullBatchTrainer
from neuro.stopping import EarlyStopping
import abalone
from neuro.weightdecay import Renormalize
import numpy
from neuronaut_plot import AnimationRender, render_confusion
import json
import time
import os
from mako.template import Template

logging.basicConfig(level=logging.INFO)

ctx = CUDAContext()

dir_name = os.path.join("reports", "report-abalone-"+ time.strftime("%Y%m%d-%H%M%S"))
vid_name = "training.webm"
os.mkdir(dir_name)

inp, targ, inpt, targt, minimum, maximum = abalone.get_patterns()
inp, targ, inpt, targt = ctx.upload(inp, targ, inpt, targt)

logging.info(inp.shape)
logging.info(targ.shape)

assert inp.dtype == inpt.dtype == numpy.float32
assert targ.dtype == targt.dtype == numpy.float32

class MyNetwork(RegressionNetwork, FeedForwardNeuralNetwork):
    pass

net = MyNetwork(context=ctx, input_shape=(8,))

class LogisticDenseLayer(Logistic, DenseLayer):
    pass

net.add_layer(LogisticDenseLayer, num_units=128)
net.add_layer(LogisticDenseLayer, num_units=128)
net.add_layer(LogisticDenseLayer, num_units=1)

ar = AnimationRender(os.path.join(dir_name, vid_name))
gr = AnimationRender(os.path.join(dir_name, "step_sizes.webm"), vmin=0.00000001, vmax=0.01, divergent=False)
cr = AnimationRender(os.path.join(dir_name, "confusion.webm"), vmin=minimum, vmax=maximum, render_function=render_confusion, array_shape=targt.shape)
ar.start()
gr.start()
cr.start()

expected = targt.get()

class MyTrainer(Renormalize, EarlyStopping, History, RPROPMinus, BackpropagationTrainer, FullBatchTrainer):
    def on_new_best(self, old_best, new_best):
        super(MyTrainer, self).on_new_best(old_best, new_best)

    def train_step(self, *args, **kwargs):
        super(MyTrainer, self).train_step(*args, **kwargs)
        if self.steps % 10 == 0: # only every 10th frame gets drawn
            ar.add_frame(self.network.download())
            gradients = [(layer.step_sizes[0].get(), layer.step_sizes[1].get()) for layer in self.training_state.layers]
            gr.add_frame(gradients)
            cr.add_frame((expected, self.test_state.layers[-1].activations.get()))

trainer = MyTrainer(network=net, training_data=(inp, targ), test_data=(inpt, targt), l2_decay=0.0001, min_steps=10000)
net.reset()
start_time = time.time()
trainer.train()
end_time = time.time()
time_taken = end_time - start_time
ar.join()
gr.join()
cr.join()

hist = trainer.errors['history']['test']
hist = zip(range(1, len(hist)*trainer.validation_frequency+1, trainer.validation_frequency), hist)

description = {
    '#Training Patterns' : inp.shape[0],
    '#Test Patterns' : inpt.shape[0],
    '#Attributes' : inp.shape[1],
    '#Targets' : targ.shape[1],
    'Task' : 'Regression'
}

layers = [{
    'Neurons' : layer.output_shape,
    'Type' : type(layer).__name__
} for layer in net.layers]

summary = {
    '#Total Steps' : len(hist),
    'Best Test Error' : trainer.errors['best']['test'],
    'Best Step' : trainer.best_step,
    'Seconds taken' : time_taken,
    'Steps/Second' : trainer.steps_per_sec
}

trainer_classes = [cls.__name__ for cls in trainer.__class__.__bases__]
data = dict(history=hist, trainer_classes=trainer_classes, dataset_name="abalone", summary=summary, layers=layers, description=description, dataset_url="https://archive.ics.uci.edu/ml/datasets/Abalone", time=time.asctime(), parameters=trainer.parameters)
data_string = json.dumps(data)

html = Template(filename="neuronaut-report.html", input_encoding='utf-8').render(data=data_string)
with open(os.path.join(dir_name, "report.html"), "wb") as f:
    f.write(html.encode('utf-8'))
