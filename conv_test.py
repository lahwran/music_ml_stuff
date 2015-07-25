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
            self.stride = max(poolamount / 3, 1)
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



from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, ConvolutionalActivation


from blocks.bricks import Rectifier
convolutions = ConvolutionalSequence(
    layers=[
        ConvolutionalLayer(
            name="c1",
            activation=Rectifier().apply,
            filter_size=(9, 1),
            num_filters=10,
            conv_step=(2, 1),
            pooling_size=(2, 1),
            pooling_step=(1, 1),
            ),
        ConvolutionalActivation(
            name="c2",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c3",
            activation=Rectifier().apply,
            filter_size=(3, 1),
            num_filters=10,
            conv_step=(1, 1),
            pooling_size=(3, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c4",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c5",
            activation=Rectifier().apply,
            filter_size=(5, 1),
            num_filters=10,
            conv_step=(1, 1),
            pooling_size=(5, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c6",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c7",
            activation=Rectifier().apply,
            filter_size=(11, 1),
            num_filters=10,
            conv_step=(4, 1),
            pooling_size=(11, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c8",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c9",
            activation=Rectifier().apply,
            filter_size=(21, 1),
            num_filters=10,
            conv_step=(7, 1),
            pooling_size=(21, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c10",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c11",
            activation=Rectifier().apply,
            filter_size=(31, 1),
            num_filters=10,
            conv_step=(10, 1),
            pooling_size=(31, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c12",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c13",
            activation=Rectifier().apply,
            filter_size=(41, 1),
            num_filters=10,
            conv_step=(13, 1),
            pooling_size=(41, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c14",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
        ConvolutionalLayer(
            name="c15",
            activation=Rectifier().apply,
            filter_size=(41, 1),
            num_filters=10,
            conv_step=(13, 1),
            pooling_size=(41, 1),
            pooling_step=(1, 1),
        ),
        ConvolutionalActivation(
            name="c16",
            activation=Rectifier().apply,
            filter_size=(1, 1),
            num_filters=1),
    ],
    image_size=(input_size, 1),
    num_channels=2)

convolutions.initialize()

