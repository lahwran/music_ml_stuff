from cfft import fft, fft_frequencies
import skimage
import skimage.io
import time
import numpy
import itertools

skimage.io.use_plugin("freeimage")

def blocks(data, block_size, block_stride):
    result = []
    samples = data.shape[0]
    spare = samples % block_size
    samples -= spare
    for begin in xrange(0, samples - block_size, block_stride):
        end = begin + block_size
        result.append(data[begin:end])
    return result

#def blocks(data, block_size):
#    # adapted from https://github.com/MattVitelli/GRUV/blob/master/data_utils/parse_files.py
#    # takes ownership of data
#
#    a = time.time()
#    total_samples = data.shape[0]
#    print "a"
#    spare = len(data) % block_size
#    print "b"
#    if spare != block_size:
#        print "c"
#        data = data[:-spare]
#        print "d"
#    data.shape = (-1, block_size)
#    print "e"
#    print "converted blocks in", time.time() - a
#    return data

def freqanalysis(data, samplerate):
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)
    assert samplerate == 44100, samplerate
    view = data.view()
    blocksize = 8192 # high resolution
    blockstride = 1024 # 43 per second
    # 11x11, stride 3
    b = blocks(view, blocksize, blockstride)
    a = time.time()
    result = numpy.array([fft(block)[:blocksize/4] for block in b])
    print "fft:", time.time() - a,
    return result, blocksize


def savefft(data, filename, windowsize):
    a = time.time()

    dd = data.transpose()
    scale = 256
    minindex = 0
    for index, freq in enumerate(fft_frequencies(xrange(windowsize), 44100)):
        if freq < 20:
            continue
        minindex = index
        break

    asdf = numpy.ceil(numpy.log2(dd.shape[0])) * scale + 1
    begin_offset = numpy.floor(numpy.log2(minindex)) * scale - 1
    log_scaled = numpy.zeros((asdf - begin_offset, dd.shape[1]))
    oldindex = None
    for index, row in enumerate(dd):
        newindex = numpy.log2(index + 1) * scale
        if index >= minindex:
            # nested to let the thing after the if run and 'cause I'm lazy
            ceil = numpy.ceil(newindex)
            ceil_lerp = ceil - newindex
            log_scaled[ceil - begin_offset,:] += row * ceil_lerp
            if oldindex is not None:
                floor = numpy.floor(oldindex)
                floor_lerp = 1 - (oldindex - floor)
                log_scaled[floor-begin_offset,:] += row * floor_lerp
                if floor < ceil:
                    for x in xrange(int(floor+1),int(ceil)):
                        log_scaled[x-begin_offset,:] += row
        else:
            # dc offset row - don't give a
            pass
        oldindex = newindex
    log_scaled /= numpy.max(log_scaled)
    log_scaled *= 8192
    log_scaled = numpy.log10(log_scaled + 1) * 16000
    skimage.io.imsave(filename, log_scaled.astype('uint16'))
    print "written:", time.time() - a,

if __name__ == "__main__":
    from audio2numpy import load, get_current_song
    data, samplerate = load(get_current_song())
    ff, windowsize = freqanalysis(data, samplerate)
    savefft(ff, "test.png", windowsize)
