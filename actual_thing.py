

import itertools
import random
import numpy
import skimage
import skimage.io
import os
import datetime
#import fuel
#import fuel.datasets

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras_models_edited import UnsupervisedWackySequential
from keras.callbacks import ModelCheckpoint

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
    print "time to load an image:", b-a

    return array


def pick_chunks(loaded_image, chunk_count=2, random_seed=None):
    if random_seed is None:
        r = random
    else:
        r = random.Random(random_seed)
    iter1 = iter(loaded_image)
    iter2 = iter(loaded_image)
    try:
        iter2.next()
    except StopIteration:
        raise Exception("Got empty loaded_image!")

    pairs = zip(iter1, iter2)
    included_pairs = r.sample(pairs, 2)
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



def train():
    random.seed(0)
    basedir = os.path.join(cache_dir, "frequency")
    paths = [x for x in os.listdir(basedir) if x.endswith("png")]
    random.shuffle(paths)
    train_length = int(len(paths) * 0.95)

    X = numpy.array(paths[:30])

    batch_size = 4 # times 12, see pick_chunks
    epochs = 2
    def loading_func(filenames, deterministic):
        inputs = []
        outputs = []
        seed = None
        if deterministic:
            seed = 0
        for filename in filenames:
            image = load_image(os.path.join(basedir, filename))
            print "getting outputs"
            for image1, image2 in pick_chunks(image, seed):
                inputs.append([image1])
                #TODO: this would probably be faster if it was shipped to the
                # GPU in a batch. predict allows this, I just didn't want to
                # do even more dimensionality-figuring-out
                output = model.predict(numpy.array([[image2]]), verbose=1)[0]
                output = output.reshape((256,))
                #output += numpy.random.normal(loc=0.05, scale=0.1, size=output.shape)
                outputs.append(output)
                print '.',
                del image2
            print
        return numpy.array(inputs), numpy.array(outputs)
    #loading_func([paths[0]], True)
    #return
    # TODO: save model each run

    model.compile(loss="mean_squared_error", optimizer="adadelta",
            loading_func=loading_func)
    #import pudb; pudb.set_trace()
    loading_func([paths[0]], True)

    #checkpointer = ModelCheckpoint('model_%s.hdf5' % datetime.datetime.now(),
    #        verbose=1, save_best_only=True)
    #checkpointer_latest = ModelCheckpoint('model_%s_latest.hdf5' % datetime.datetime.now(),
    #        verbose=1)
    #model.fit(X, callbacks=[checkpointer, checkpointer_latest],
    #        verbose=1)


if __name__ == "__main__":
    train()
