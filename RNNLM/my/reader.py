# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import numpy as np
from numpy import *

Py3 = sys.version_info[0] == 3
length = 0
sequence_length = []
word_to_id = {}

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      a = f.read().strip().split("\n")
      global length
      length = len(a)
      b = []
      for line in a:
        b = b+line.split()+["\n"]
      return b
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()

def get_dict():
  global word_to_id
  word_to_id = eval(open("./cha_to_id.txt").read())
  return word_to_id

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  global word_to_id
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  #print(len(word_to_id))
  return word_to_id


def _file_to_word_ids(filename):
  data = _read_words(filename)
  global word_to_id
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, is_training = True, index=0):
  #valid_path = os.path.join(data_path, "ptb.valid.txt")

  """
  global word_to_id
  word_to_id = eval(open("./cha_to_id.txt").read())
  if is_training == True:
    train_path = os.path.join(data_path, "Total_cha.txt")
    #word_to_id = _build_vocab(train_path)
    #f = open("./cha_to_id.txt","w")
    #f.write(str(word_to_id))
    #f.close()
    train_data = _file_to_word_ids(train_path)
  else:
    test_path = os.path.join(data_path, "test_char.txt")
    train_data = _file_to_word_ids(test_path)
  
  #vocabulary = len(word_to_id)
  ind = word_to_id["\n"]
  result = []
  tem = []
  co = 0
  global sequence_length
  for i in train_data:
    if i==ind:
      sequence_length.append(len(tem)-2)
      for temi in range(47-co):
        tem += [len(word_to_id)]
      result+=tem
      tem = []
      co = 0
      continue
    co += 1
    tem.append(i)
  """
  print("Getting Data...")
  global word_to_id
  word_to_id = eval(open("./cha_to_id.txt").read())
  if is_training == True:
    if(index<10):
      index = "0"+str(index)
    #train_path = os.path.join(data_path, "conv.txt")
    #train_path = os.path.join(data_path, "corpus_cha.txt")
    #train_path = os.path.join(data_path, "pro_cha.txt")
    train_path = os.path.join(data_path, "corpus/total_"+str(index))
    train_data = array(open(train_path).read().strip().replace("\n"," ").split(),dtype=int32)
    
    #word_to_id = _build_vocab(train_path)
    #f = open("./cha_to_id.txt","w")
    #f.write(str(word_to_id))
    #f.close()
    
    #data_path = os.path.join(data_path,"conv_length.txt")
    data_path = os.path.join(data_path,"length/length_"+str(index))
    #data_path = os.path.join(data_path,"corpus_cha_length.txt")
    #data_path = os.path.join(data_path,"pro_cha_length.txt")
    sequence_length = open(data_path).read().strip().split()

    if len(sequence_length)<=1:
      sequence_length =  open(data_path).read().strip().split("\n")
    sequence_length = array(sequence_length, dtype = int32)

    print("length before filter: %d"%shape(sequence_length)[0])

    if len(train_data)%47 != 0:
      train_data = array([i.split() for i in open(train_path).read().strip().split("\n")])
      tem_index = []
      for i in range(len(train_data)):
        if len(train_data[i]) != 47:
          tem_index.append(i)
      print("length deleted index: %d"%(len(tem_index)))
      train_data = delete(train_data, tem_index)
      train_data = array(concatenate(train_data),dtype=int32)
      sequence_length = delete(sequence_length, tem_index)
      
    print("length after filter: %d"%shape(sequence_length)[0])


  else:
    #test_path = os.path.join(data_path, "")
    #test_path = os.path.join(data_path, "test_normal.txt")
    global length
    sequence_length = []
    test_data = open(data_path).read().strip().split("\n")
    length = len(test_data)
    train_data = ""
    for line in test_data:
      line = line.split()
      temlen = len(line)
      if temlen>47:
        continue
      tems = ""
      sequence_length.append(temlen)
      for word in line:
        tems += str(word_to_id[word])+" "
      tems += "9173 "*(47-temlen)
      train_data +=tems
    train_data = array(train_data.strip().split(),dtype=int32)
    print(shape(train_data))

    sequence_length = array(sequence_length, dtype=int32)
  print("Getting Data Finish")
  return train_data, sequence_length


def ptb_producer(raw_data,sequence_length, batch_size, num_steps, name=None):
  print("Producing batch...")
  with tf.name_scope(name, "PTBProducer", [raw_data, sequence_length, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    sequence_length = tf.convert_to_tensor(sequence_length, name="sequence_length", dtype=tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    sequence_length = tf.reshape(sequence_length[0 : batch_size * (batch_len//num_steps)],[batch_size,batch_len//num_steps])
    epoch_size = (batch_len) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=True).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])

    seq_length = tf.strided_slice(sequence_length, [0, i],[batch_size, i+1])
    seq_length.set_shape([batch_size,1])
    
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    print(y)
    print("Producing batch finish")
    return x, y, seq_length