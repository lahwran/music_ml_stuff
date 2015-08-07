

from functools import partial
import traceback
import itertools
import random
import numpy
import skimage
import skimage.io
import os
import datetime
#import fuel
#import fuel.datasets

import theano
import theano.compile
import theano.compile.monitormode
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras_models_edited import UnsupervisedWackySequential
from keras.callbacks import ModelCheckpoint
import keras.constraints

from localdata import cache_dir
from freq import blocks
skimage.io.use_plugin("freeimage")

def load_image(path):
    import time
    a = time.time()
    array = skimage.io.imread(path)
    array = array.transpose()[200:,:]
    array = blocks(array, 123, 123)
    b = time.time()

    return array


# can't handle 15
# can handle 10
def pick_chunks(loaded_image, random_seed=None, chunk_count=3):
    if random_seed is None:
        r = random
    else:
        r = random.Random(random_seed)
    iter1 = iter(loaded_image)
    iter2 = iter(loaded_image)
    try:
        iter2.next()
    except StopIteration:
        return []

    pairs = zip(iter1, iter2)
    if len(pairs) < 1:
        return []
    included_pairs = r.sample(pairs, min(chunk_count, len(pairs)))
    swapped_pairs = (r.sample(pair, 2) for pair in included_pairs)
    copied_pairs = [(numpy.copy(a), numpy.copy(b)) for a, b in swapped_pairs]
    return copied_pairs


def get_files(directory=None):
    # no minibatches to be found here
    if directory is None:
        directory = os.path.join(cache_dir, "frequency")
    files = [x for x in os.listdir(directory) if x.endswith('png')]
    random.shuffle(files)
    return files

def iterate_whole_dataset():
    for file in get_files():
        for block in load_image(file):
            yield block

def split_seq(iterable, size):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def iterate_minibatches(batch_size):
    return split_seq(iterate_whole_dataset(), batch_size)



#class FrequencyImages(fuel.datasets.Dataset):
#    def __init__(self, path, preprocess=None):
#
#        self.preprocess = preprocess
#        super(TextFile, self).__init__()
#
#    def open(self):
#        return chain(*[iter_(open(f)) for f in self.files])
#
#    def get_data(self, state=None, request=None):
#        if request is not None:
#            raise ValueError
#        sentence = next(state)
#        if self.preprocess is not None:
#            sentence = self.preprocess(sentence)
#        data = [self.dictionary[self.bos_token]] if self.bos_token else []
#        if self.level == 'word':
#            data.extend(self.dictionary.get(word,
#                                            self.dictionary[self.unk_token])
#                        for word in sentence.split())
#        else:
#            data.extend(self.dictionary.get(char,
#                                            self.dictionary[self.unk_token])
#                        for char in sentence.strip())
#        if self.eos_token:
#            data.append(self.dictionary[self.eos_token])
#        return (data,)



# 123 pixels wide
# 1845 pixels tall
# stride of 45
# possible: 10x 9x9 pad 9 stride 3 -> /2 -> 10x 9x9 pad stride 3 -> 10x 8x8 -> 1x
# possible: 9 + 3(x-1) = 123; 3 + 2(y-1) = x; 3 + 2(z-1) = y;
#        -> 10x 9x9 stride 3 -> 39x..; 10x 3x3 stride 2 -> 19x..; 10x 3x3 stride 2 -> 9x..; 10x 9x9 -> 1x..; vertical conv from here
# possible: 9 + 3(x-1) = 123; 13 + 3(y-1) = x/3


model = UnsupervisedWackySequential()

Convolution2D = partial(Convolution2D, W_constraint=keras.constraints.maxnorm(2))
Dense = partial (Dense, W_constraint=keras.constraints.maxnorm(2))

# this was a default configuration for keras. seems as good as anything to start with.
model.add(Convolution2D(16, 1, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(16, 16, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 16, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(64, 64, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 64, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(128, 128, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 128, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(256, 256, 3, 3, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(512, 256, 2, 2, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(Convolution2D(64, 512, 2, 2, border_mode="full", init="he_normal"))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(1, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(61568, 256, init="he_normal"))
model.add(Activation("relu"))
model.add(Dropout(0.25))


def shift(wrap_array, delta=0.1):
    wrap_array += numpy.sign([wrap_array[x] - wrap_array[x-1] for x in range(wrap_array.shape[0])]) * delta

exceptioncount = 0

def train(load_from=None):
    global exceptioncount
    random.seed(1)
    basedir = os.path.join(cache_dir, "frequency")
    paths = [x for x in os.listdir(basedir) if x.endswith("png")]
    random.shuffle(paths)
    skips = set(open("badlist", "r").read().replace("/mnt/frequency/", "").split('\n'))
    paths = [x for x in paths if x not in skips]
    train_length = int(len(paths) * 0.95)

    X = numpy.array(paths[:3000])
    X_valid = numpy.array(paths[3000:3300])

    def loading_func(filenames, deterministic):
        # TODO: introducing hyperparameters!? this is going to fail horribly.
        # if I had more mental space right now, I'd do this so that shift used
        # scaling rather than shifting (multiplication instead of addition).
        # but my brain is full and I'm on a deadline, so the worst case is that
        # this turns out to be another failed attempt.
        hyper_normal_loc = 0.00
        hyper_normal_scale = 0.1
        hyper_in_song_delta = 0.01
        hyper_batch_delta = 0.1

        global exceptioncount
        inputs = []
        batch_song_outputs = []
        seed = None
        if deterministic:
            seed = 1
        for filename in filenames:
            song_outputs = []
            try:
                image = load_image(os.path.join(basedir, filename))
                for image1, image2 in pick_chunks(image, seed):
                    #TODO: this would probably be faster if it was shipped to the
                    # GPU in a batch. predict allows this, I just didn't want to
                    # do even more dimensionality-figuring-out
                    output = model.predict(numpy.array([[image2]]))[0]

                    # normalize
                    output = output.reshape((256,))
                    output -= numpy.min(output)
                    m = numpy.max(output)
                    output /= m
                    output -= 0.5
                    output *= 2

                    # tweak
                    output += numpy.random.normal(loc=hyper_normal_loc,
                            scale=hyper_normal_scale,
                            size=output.shape)


                    inputs.append([image1])
                    song_outputs.append(output)
                    del image2
            except KeyboardInterrupt:
                raise
            except:
                print "error loading", filename
                with open("errors", "wa") as writer:
                    writer.write("\n")
                    writer.write("filename: ")
                    writer.write(filename)
                    writer.write("\n")
                    writer.write(traceback.format_exc())
                    writer.write('\n')

                exceptioncount += 1
                if exceptioncount > 12000:
                    raise
                else:
                    traceback.print_exc()
                    continue
            # shift in-song values away from each other just slightly
            song_outputs = numpy.array(song_outputs)
            shift(song_outputs, delta=hyper_in_song_delta)
            batch_song_outputs.append(song_outputs)

        # shift song values away from each other by a fairly large margin
        batch_song_outputs = numpy.array(batch_song_outputs)
        shift(batch_song_outputs, delta=hyper_batch_delta)
        #numpy.clip(batch_song_outputs, -1, 1, out=batch_song_outputs)

        result = numpy.array(inputs), batch_song_outputs.reshape(-1, 256)
        print result[0].shape
        print result[1].shape
        return result
    model.compile(loss="mean_squared_error", optimizer="adadelta",
            loading_func=loading_func)
    print loading_func(paths[0:3], True)[1]
    ##return

    #asdf = "/tmp/weights"
    #model.save_weights(asdf)
    #model.load_weights(asdf)
    #print loading_func([paths[0]], True)[1][0]
    #raise SystemExit
    if load_from is not None:
        model.load_weights(load_from)


    checkpointer = ModelCheckpoint('model_%s.hdf5' % datetime.datetime.now(),
            verbose=1, save_best_only=True)
    checkpointer_latest = ModelCheckpoint('model_%s_latest.hdf5' % datetime.datetime.now(),
            verbose=1)
    # default number of epochs = 100
    print model.evaluate(X[:3], show_accuracy=True)
    try:
        model.fit(X, callbacks=[checkpointer, checkpointer_latest],
                verbose=1, batch_size=3, nb_epoch=40000, validation_data=X_valid,
                show_accuracy=True)
    finally:
        image = pick_chunks(load_image(os.path.join(basedir, X[-1])), True)
        output = model.predict(numpy.array([[image[0][1]]]))[0]
        print output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", default=None, nargs="?")
    args = parser.parse_args()
    train(load_from=args.file)
