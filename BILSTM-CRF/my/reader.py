import collections
import os
import sys
from copy import deepcopy
import tensorflow as tf
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from numpy import *

Py3 = sys.version_info[0] == 3
length = 0
sequence_length = []
word_to_id = {}
similar = eval(open("./similarList.txt").read())

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
    train_path = os.path.join(data_path, "conv.txt")

    #train_path = os.path.join(data_path, "corpus_cha.txt")
    #train_path = os.path.join(data_path, "pro_cha.txt")
    train_path = os.path.join(data_path, "corpus/total_"+str(index))
    train_data = array(open(train_path).read().strip().replace("\n"," ").split(),dtype=int32)
    
    #word_to_id = _build_vocab(train_path)
    #f = open("./cha_to_id.txt","w")
    #f.write(str(word_to_id))
    #f.close()
    data_path = os.path.join(data_path,"length/length_"+str(index))
    #data_path = os.path.join(data_path,"conv_length.txt")
    #data_path = os.path.join(data_path,"corpus_cha_length.txt")
    #data_path = os.path.join(data_path,"pro_cha_length.txt")
    sequence_length = open(data_path).read().strip().split()
    if len(sequence_length)<=1:
      sequence_length =  open(data_path).read().strip().split("\n")
    sequence_length = array(sequence_length, dtype = int32)
    print(shape(sequence_length))
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
      train_data += tems
    train_data = array(train_data.strip().split(),dtype=int32)
    sequence_length = array(sequence_length, dtype=int32)

  print("Getting Data Finish")
  if is_training == False:
    return train_data, sequence_length

  else:
    train_data = train_data.reshape(-1,47)
    trainx, devx, trainlength, devlength = train_test_split(train_data, sequence_length, shuffle = True, test_size = 0.02)
    return reshape(array(trainx),[-1]), trainlength, reshape(array(devx),[-1]), devlength


def ptb_producer(raw_data,sequence_length, batch_size, num_steps, name=None, is_training = True, test_path = None):
  if is_training == True:
    raw_data = reshape(array(raw_data,dtype = int32),[-1, num_steps])
    X = []
    y = []
    print("Creating Sequences...")
    print(shape(raw_data)[0])
    for i in range(1):
      for line in range(shape(raw_data)[0]):
        linelength = sequence_length[line]
        xline = raw_data[line]
        yline = [[0,1] for i in range(num_steps)]

        randmod = randint(0,8)
        if randmod<=2:
          pass 
        elif randmod<=6:
          randnum = randint(1,linelength-2)
          co = 0
          while ( (xline[randnum] not in similar)  or (len(similar[xline[randnum]])<2)) and co<20:
            randnum = randint(1,linelength-2)
            co += 1
          if co<20:
            yline[randnum] = [1,0]
            xline[randnum] = similar[xline[randnum]][randint(0,len(similar[xline[randnum]])-1)] 
        else:
          rand1 = randint(1, linelength-2)
          rand2 = randint(1, linelength-2)

          co = 0
          while ( (xline[rand1] not in similar) or (len(similar[xline[rand1]])<2) ) and co<20:
            rand1 = randint(1,linelength-2)
            co += 1
          if co < 20:
            yline[rand1] = [1,0]
            xline[rand1] = similar[xline[rand1]][randint(0,len(similar[xline[rand1]])-1)] 

          co = 0
          while ( (xline[rand2] not in similar) or (rand2 == rand1) or (len(similar[xline[rand2]])<2) ) and co<20:
            rand2 = randint(1,linelength-2)
            co += 1
          if co < 20:
            yline[rand2] = [1,0]
            xline[rand2] = similar[xline[rand2]][randint(0,len(similar[xline[rand2]])-1)] 
        X.append(xline)
        y.append(yline)

    print("sequences length:%d"%len(y))
    X = array(X).reshape(-1,1)
    y = array(y).reshape(-1,2)
    
    print(shape(X))  
    print(shape(y))  

  else:
    X = array(raw_data).reshape(-1,1)
    f = open(test_path+"_ans","r").read().strip().split("\n")
    y = [[[0,1] for _ in range(47)] for i in range(shape(X)[0]//47) ]
    for line in range(len(f)):
      fline = f[line].split()
      if fline[0]=="-1":
        continue
      else:
        for i in fline:
          y[line][int(i)] = [1,0]
    y = reshape(y,[-1,2])
    print(shape(X))


  print("Producing batch...")
  with tf.name_scope(name, "PTBProducer", [X,y, batch_size, num_steps]):
    print(shape(X))
    print(shape(y))
    x = tf.convert_to_tensor(X, name="x", dtype=tf.int32)
    y = tf.convert_to_tensor(y, name="y", dtype=tf.int32)
    sequence_length = tf.convert_to_tensor(sequence_length, name="sequence_length", dtype=tf.int32)
    
    data_len = tf.size(x)
    batch_len = data_len // batch_size
    x = tf.reshape(x[0 : batch_size * batch_len],[batch_size,batch_len])
    y = tf.reshape(y[0 : batch_size * batch_len],[batch_size,batch_len,2])
    sequence_length = tf.reshape(sequence_length[0 : batch_size * (batch_len//num_steps)],[batch_size,batch_len//num_steps])
    epoch_size = (batch_len) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")

    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=True).dequeue()
    x = tf.strided_slice(x, [0, i * num_steps],[batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])

    seq_length = tf.strided_slice(sequence_length, [0, i],[batch_size, i+1])
    seq_length.set_shape([batch_size,1])
    
    y = tf.strided_slice(y, [0, i * num_steps,0],[batch_size, (i + 1) * num_steps,2])
    y.set_shape([batch_size, num_steps,2])
    print("Producing batch finish")
    print(x)
    #print(y)
    return x, y, seq_length