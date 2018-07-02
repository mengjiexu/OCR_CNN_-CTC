# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Definitions that don't fit elsewhere.

"""
import glob
import numpy

import cv2
import numpy as np

# Constants
import time

SPACE_INDEX = 0
# FIRST_INDEX = ord('0') - 1  # 0 is reserved to space
FIRST_INDEX = 1  # 0 is reserved to space

SPACE_TOKEN = '<space>'

__all__ = (
    'DIGITS',
    'sigmoid',
    'softmax',
    'CHARS'
)


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CHARS = list(DIGITS + LETTERS)

LENGTHS = [6, 6]  # the number of digits varies from LENGTHS[0] to LENGTHS[1] in a image
TEST_SIZE = 100
ADD_BLANK = True  # if add a blank between digits
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000

# parameters for bdlstm ctc
BATCH_SIZE = 64
BATCHES = 100

OUTPUT_SHAPE = (BATCH_SIZE, 256)

TRAIN_SIZE = BATCH_SIZE * BATCHES

MOMENTUM = 0.9
REPORT_STEPS = 1000

# Hyper-parameters
num_epochs = 2000
num_hidden = 128
num_layers = 2

# Some configs
# Accounting the 0th indice +  space + blank label = 28 characters
# num_classes = ord('9') - ord('0') + 1 + 1 + 1
num_classes = len(CHARS) + 1 + 1  # 10 digits + blank + ctc blank
print(num_classes)


def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]


def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))


"""
    {"dirname":{"fname":(im,code)}}
"""
data_set = {}


def load_data_set(dirname):
    fname_list = glob.glob(dirname + "/*.png")
    result = dict()
    print("loading", dirname)
    for fname in sorted(fname_list):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = list(fname.split("/")[1].split("_")[1])
        index = fname.split("/")[1].split("_")[0]
        result[index] = (im, code)
    data_set[dirname] = result


def read_data_for_lstm_ctc(dirname, start_index=None, end_index=None):
    start = time.time()
    fname_list = []
    if dirname not in data_set.keys():
        load_data_set(dirname)

    if start_index is None:
        f_list = glob.glob(dirname + "/*.png")
        fname_list = [fname.split("/")[1].split("_")[0] for fname in f_list]

    else:
        for i in range(start_index, end_index):
            fname_index = "{:08d}".format(i)
            # print(fname_index)
            fname_list.append(fname_index)
    # print("regrex time ", time.time() - start)
    start = time.time()
    dir_data_set = data_set.get(dirname)
    for fname in sorted(fname_list):
        # im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        # code = list(fname.split("/")[1].split("_")[1])
        im, code = dir_data_set.get(fname)
        yield im, numpy.asarray(
            [SPACE_INDEX if x == SPACE_TOKEN else (CHARS.index(x) + FIRST_INDEX) for x in list(code)])
        # print("get time ", time.time() - start)


#        print numpy.asarray([SPACE_INDEX if x == SPACE_TOKEN else (CHARS.index(x) + FIRST_INDEX) for x in list(code)])


def convert_original_code_train_code(code):
    return numpy.asarray([SPACE_INDEX if x == SPACE_TOKEN else (CHARS.index(x) - FIRST_INDEX) for x in code])


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


if __name__ == '__main__':
    train_inputs, train_codes = unzip(list(read_data_for_lstm_ctc("test"))[:2])
    print(train_inputs.shape)
    print(train_codes)
    print("train_codes", train_codes)
    targets = np.asarray(train_codes).flat[:]
    print(targets)
    print(list(read_data_for_lstm_ctc("test", 0, 10)))
