from scikits import audiolab
from matplotlib import pyplot
import numpy
import subprocess
import mpd
import os
from localdata import mpd_password as password
from localdata import cache_dir
from localdata import metadata
import hashlib

lengths = sorted([m["len"] for m in metadata.values()])
input_size = sorted(lengths)[5:-5][0]

devnull = open("/dev/null", "w")

class SkipHack(Exception):
    pass


def load(path, skip=False):
    # note: contrary to previous assumption, this returns stereo if stereo is
    # available. whoops!
    cached = os.path.join(cache_dir, "audio", hashlib.sha256(path.encode("utf-8")).hexdigest() + ".wav")

    import time
    z = time.time()
    if not os.path.exists(cached):
        print "converting", path,
        subprocess.check_call(["ffmpeg", "-y", "-i", path, "-osr", "44100",
                cached],
                stdout=devnull, stderr=devnull)
        if skip:
            print "converted file in ", time.time() - z
    elif skip:
        print "skipped."

    stat = os.stat(cached)
    if stat.st_size > 70 * 1024 * 1024:
        print "SKIPPING %smb" % (stat.st_size / (1024 * 1024))
        raise SkipHack
    else: print "DOING %smb" % (stat.st_size / (1024 * 1024))

    if not skip:
        data, samplerate, format_ = audiolab.wavread(cached)
        print "loaded:", time.time() - z,
        return data, samplerate


def convert_to_floatarray(path):
    cached_wav = os.path.join(cache_dir, "audio", hashlib.sha256(path.encode("utf-8")).hexdigest() + ".wav")
    cached_numpy = os.path.join(cache_dir, "numpy", hashlib.sha256(path.encode("utf-8")).hexdigest() + ".npy")
    if os.path.exists(cached_numpy):
        print "exists, skipping"
        return cached_numpy

    data, samplerate = load(path)

    if len(data) < input_size:
        print "skipping due to being too small:", path
        return

    middle = len(data)/2
    half = input_size / 2
    begin = middle - half
    end = middle + half
    cut = data[begin:end]
    del data

    numpy.save(cached_numpy, cut)

    os.unlink(cached_wav)

    return cached_numpy


def convert_to_png_freq(path):
    cached_wav = os.path.join(cache_dir, "audio", hashlib.sha256(path.encode("utf-8")).hexdigest() + ".wav")
    cached_png = os.path.join(cache_dir, "frequency", hashlib.sha256(path.encode("utf-8")).hexdigest() + ".png")
    if os.path.exists(cached_png):
        print "s",
        return cached_png

    data, samplerate = load(path)
    if samplerate != 44100:
        return None

    import freq
    ff, aa = freq.freqanalysis(data, samplerate)
    freq.savefft(ff, cached_png, aa)

    os.unlink(cached_wav)

    return cached_png
    

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
