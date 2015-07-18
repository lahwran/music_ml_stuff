
import numpy

#windows =

numpy.set_printoptions(precision=3, linewidth=119)

def fft(data):
    return numpy.absolute(numpy.fft.rfft(data))

def fft_frequencies(data, samplerate):
    return numpy.fft.fftfreq(len(data), 1.0 / samplerate)

for window in range(7, 15):
    samples = 2 ** window

    n = numpy.fft.fftfreq(samples, 1.0 / 44100)
    nn = n[1:len(n)/2]
    print "lowfreq:", min(nn), "highfreq:", max(nn), "step:", nn[1] - nn[0], "bins:", len(nn), "wps:", 44100.0/samples
