from scikits import audiolab
from matplotlib import pyplot
import numpy
import subprocess
import mpd
import os
from localdata import mpd_password as password


devnull = open("/dev/null", "w")


def load(path):
    import time
    z = time.time()
    subprocess.call(["ffmpeg", "-y", "-i", path, "/tmp/tmp.wav"],
            stdout=devnull, stderr=devnull)

    data, samplerate, format_ = audiolab.wavread("/tmp/tmp.wav")
    print "loaded file in", time.time() - z
    return data, samplerate


def get_current_song():
    client = mpd.MPDClient()

    client.connect("localhost", 6600)
    client.password(password)

    currentsong = client.currentsong().get("file")
    path = os.path.join(os.path.expanduser("~/mpd/"), currentsong)
    return path


def display_song(data):
    #h, y = numpy.histogram(data, 10000)
    windowsize = 8192
    offset = len(data)/2 - windowsize/2
    end = offset+windowsize


    h = numpy.absolute(numpy.fft.rfft(data[offset:end]))
    h = numpy.absolute(numpy.fft.rfft(h[:len(h)/2]))

    pyplot.plot(data[offset:end])
    pyplot.show()


def main():
    data, samplerate = load(get_current_song())
    display_song(data)

if __name__ == "__main__":
    main()
