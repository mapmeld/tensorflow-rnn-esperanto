# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sys import argv
import tensorflow as tf
import numpy as np
import my_txtutils

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

author = 'checkpoints/rnn_train_1512576159-60000000'
ncnt = 0

startPhrase = argv[1]
SPACE_CHAR = my_txtutils.convert_from_alphabet(ord(' '))

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1512576159-60000000.meta')
    new_saver.restore(sess, author)
    x = []
    x.append(my_txtutils.convert_from_alphabet(ord("M")))
    x = np.array([x])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    for i in range(len(startPhrase)):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

        if (i < len(startPhrase)):
            nextChar = my_txtutils.convert_from_alphabet(ord(startPhrase[i]))
        else:
            nextChar = SPACE_CHAR
        c, probabilityScore = my_txtutils.sample_from_probabilities(yo, topn=2, nextChar=nextChar)
        if (i < len(startPhrase)):
            c = my_txtutils.convert_from_alphabet(ord(startPhrase[i]))
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if (probabilityScore == -1):
            print (" = 1")
        else:
            print(" = " + chr(my_txtutils.convert_to_alphabet(probabilityScore)))

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0
