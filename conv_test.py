import theano
from blocks.bricks.conv import Convolutional
from audio2numpy import load, get_current_song
import audiotags
nonlyrical = audiotags.getbytag("nonlyrical")
lyrical = audiotags.getbytag("lyrical")
semilyrical = audiotags.getbytag("semilyrical")
anylyrical = nonlyrical | lyrical | semilyrical
from audiotags import basedir
from localdata import metadata

lengths = sorted([m["len"] for m in metadata.values()])
import numpy

inputwidth = 280000
print lengths[0] / float(inputwidth)
9 * 9 * inputwidth/2
avg_length = numpy.average(lengths)
input_size = sorted(lengths)[5:-5][0]

class L(object):
    def __init__(self, convcount, width, poolamount, stride=None):
        self.convcount = convcount
        self.width = width
        if stride is None:
            self.stride = max(int(poolamount / 3), 1)
        else:
            self.stride = stride
        self.poolamount = poolamount

    def outputs_size(self, input_width):
        applications_per_conv = input_width / self.stride
        ret = self.convcount * applications_per_conv / self.poolamount, self.convcount
        return ret
    
    def params_count(self, input_width, channel_count):
        parameters_per_conv = channel_count * self.width + 1
        z = self.convcount * parameters_per_conv
        return z
    
    def ops_count(self, input_width, channel_count):
        applications_per_conv = input_width / self.stride
        parameters_per_conv = channel_count * self.width
        return applications_per_conv * parameters_per_conv * self.convcount                 + applications_per_conv * 2


class FC(object):
    def __init__(self, width):
        self.width = width

    def outputs_size(self, inputs_count):
        return self.width, 1

    def params_count(self, inputs_count, channel_count):
        return inputs_count * channel_count * self.width + self.width
    
    def ops_count(self, inputs_count, channel_count):
        return inputs_count * channel_count * self.width + self.width * 2


net = [
    L(10, 9, 2, 2),
    L(1, 1, 1, 1),
    L(10, 3, 3),
    L(1, 1, 1, 1),
    L(10, 5, 5),
    L(1, 1, 1, 1),
    L(10, 11, 11),
    L(1, 1, 1, 1),
    L(10, 21, 21),
    L(1, 1, 1, 1),
    L(10, 31, 31),
    L(1, 1, 1, 1),
    L(10, 41, 41),
    L(1, 1, 1, 1),
    L(10, 41, 41),
    L(1, 1, 1, 1),
    FC(128),
    FC(2)  # cut me off and replace me when respecializing
]
ops = 0

params = 0
import datetime
layer_in = (input_size, 2)
print "input:", layer_in
print "in seconds:", datetime.timedelta(seconds=layer_in[0] / 44100.0)
print "average length:", avg_length
print "in seconds:", datetime.timedelta(seconds=avg_length/44100.0)
import math
print "square image size that would be this big:", math.sqrt(layer_in[0])
for layer in net:
    ops += layer.ops_count(*layer_in)
    pc = layer.params_count(*layer_in)
    pc1 = layer.params_count(layer_in[0], 1)
    params += pc
    
    
    layer_in = layer.outputs_size(layer_in[0])
    print "out width:", layer_in, "params:", pc

import locale
locale.setlocale(locale.LC_ALL, 'en_US')
sflops = int(722.7e9)
fflops = int(5e12)

print "total ops to run the network once:", locale.format("%d", ops, grouping=True)
print "gpu peak flops:", fflops
print "(hopefully) network runs/second on slow gpu:", float(sflops)/float(ops)
print "(hopefully) network runs/second on fast gpu:", float(fflops)/float(ops)
print "param count:", locale.format("%d", params, grouping=True)

print len(net)



from blocks.bricks.conv import (ConvolutionalLayer, ConvolutionalSequence,
        ConvolutionalActivation, Flattener)
from blocks.bricks import MLP, Softmax, FeedforwardSequence, Initializable
from blocks.initialization import Constant, IsotropicGaussian


from blocks.bricks import Rectifier


class Derpnet(FeedforwardSequence, Initializable):
    def __init__(self, input_size, **kwargs):
        self.layers = []
        self.layerinfo = [
            L(10, 9, 2, 2),
            L(10, 3, 3),
            L(10, 5, 5),
            L(10, 11, 11),
            L(10, 21, 21),
            L(10, 31, 31),
            L(10, 41, 41),
            L(10, 41, 41),
        ]
        for index, l in enumerate(self.layerinfo):
            self.layers.append(ConvolutionalLayer(
                name="conv-" + str(index * 2),
                activation=Rectifier().apply,
                filter_size=(l.width, 1),
                num_filters=l.convcount,
                conv_step=(l.stride, 1),
                pooling_size=(l.poolamount, 1),
                pooling_step=(1, 1),
                biases_init=Constant(0),
                weights_init=IsotropicGaussian(0.01),
            ))
            self.layers.append(ConvolutionalActivation(
                    name="conv-reduce-" + str(index * 2 + 1),
                    activation=Rectifier().apply,
                    filter_size=(1, 1),
                    biases_init=Constant(0),
                    weights_init=IsotropicGaussian(0.01),
                    num_filters=1
            ))
        self.convolutions = ConvolutionalSequence(
            layers=self.layers,
            image_size=(input_size, 1),
            num_channels=2)

        self.mlp_activations = [Rectifier(), Softmax()]
        self.mlp_dims = [128, 3]

        self.top_mlp = MLP(
            activations=self.mlp_activations,
            dims=self.mlp_dims,
                    # note: input is added in _push_allocation_config,
                    # unlike normal for the mlp class
            weights_init=IsotropicGaussian(0.01),
            biases_init=Constant(1))

        self.flattener = Flattener()

        super(Derpnet, self).__init__([
            self.convolutions.apply,
            self.flattener.apply,
            self.top_mlp.apply
        ], **kwargs)

    def _push_allocation_config(self):
        self.convolutions._push_allocation_config()
        conv_out_dim = self.convolutions.get_dim("output")

        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.mlp_dims

print input_size

from theano import tensor

#TODO: dropout?
#TODO: maxout?
#final_layers = MLP(

net = Derpnet(input_size)

net.initialize()

x = tensor.tensor4("x")
y = tensor.lmatrix("targets")

probs = convnet.apply(x)
cost = named_copy(CategoricalCrossEntropy().apply(y.flatten(), probs), 'cost')
error_rate = named_copy(MisclassificationRate().apply(y.flatten(), probs),
    'error_rate')

cg = ComputationGraph([cost, error_rate])

algorithm = AdaDelta(cost=cost, parameters=cg.parameters)

print net.children[0].get_dim("input_")

for layer in net.layers:
    print layer.get_dim("output")

#convolutions.initialize()
#import pudb; pudb.set_trace()
#final_layers.push_initialization_config()
#final_layers.initialize()
#x = tensor.row('noise')

#result = final_layers.apply(convolutions.apply(x))

print input_size
