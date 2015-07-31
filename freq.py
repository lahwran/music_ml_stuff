from cfft import fft
import png
import time
import numpy

def blocks(data, block_size):
    # adapted from https://github.com/MattVitelli/GRUV/blob/master/data_utils/parse_files.py
    # takes ownership of data

    a = time.time()
    total_samples = data.shape[0]
    print "a"
    spare = len(data) % block_size
    print "b"
    if spare != block_size:
        print "c"
        data = data[:-spare]
        print "d"
    data.shape = (-1, block_size)
    print "e"
    print "converted blocks in", time.time() - a
    return data

def freqanalysis(data, samplerate):
    assert samplerate == 44100
    print "taking data view"
    view = data.view()
    print "getting blocks"
    b = blocks(view, 4096)
    print "doing fft"
    a = time.time()
    result = numpy.array([fft(block)[:1024] for block in b])
    print result.shape
    print "did fft in", time.time() - a
    return result


def savefft(data, filename):
    print numpy.amin(data)
    print numpy.amax(data)
    print "writing"
    a = time.time()
    with open(filename, "wb") as writer:
        pngwriter = png.Writer(*data.shape, greyscale=True, bitdepth=16,
                compression=9)
        dd = numpy.log2(data.transpose() + 1) * 5461
        pngwriter.write(writer, dd)
    print "written in", time.time() - a

if __name__ == "__main__":
    from audio2numpy import load, get_current_song
    loaded = load(get_current_song())
    ff = freqanalysis(*loaded)
    savefft(ff, "test.png")
