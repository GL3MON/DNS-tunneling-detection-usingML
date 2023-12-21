from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import math


train_data = pd.read_csv("/content/drive/MyDrive/training.csv", encoding = 'utf-8')
train_data = train_data.sample(frac=1)

Y = train_data["1"] ## This doesnt makes sense know because the dataset is messed up
X = train_data["q+Z8AnwaBA.hidemyself.org."] ##The dataset is messed up... Try and fix it by giving it titles and changing this part of the code.

X = list(X)
Y = list(Y)

# splitting dataset
x_train = X[:int((1-0.1)*len(Y))]
x_test = X[int((1-0.1)*len(Y)):]

y_train = Y[:int((1-0.1)*len(Y))]
y_test = Y[int((1-0.1)*len(Y)):]

#x_test[10]

tokenizer = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

optim = tf.keras.optimizers.Adam(lr=1e-4)
ff = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(96,)),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

ff.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc'])

def count_vector(url) -> list:
  tmp = []
  for i in range(96):
    tmp.append(0)
  for i in url:
    if (i in tokenizer):
      tmp[tokenizer.index(i)] += 1
  return tmp

def entropy_calculator(url) -> float:
  if not url:
        return 0
  entropy = 0
  for x in range(256):
      p_x = float(url.count(chr(x)))/len(url)
      if p_x > 0:
          entropy += - p_x*math.log(p_x, 2)
  return entropy

def calculate_length(url) -> int:
  if not url:
    return 0
  return len(url)

X_Train , Y_Train = [], []
for i in range(len(y_train)):
  X_Train.append([0.,]*96)
  Y_Train.append(0)

X_Test , Y_Test = [], []
for i in range(len(y_test)):
  X_Test.append([0.,]*96)
  Y_Test.append(0)

remIndex = list()

# Making Training dataset
for i in range(len(y_train)):
  try:
    temp = count_vector(X[i])
    temp[94] = float(entropy_calculator(X[i]))
    temp[95] = float(calculate_length(X[i]))
    Y_Train[i] = y_train[i]
    X_Train[i] = temp

  except ValueError:
    remIndex.append(i)

for i in remIndex: ##removing bad data
   X_Train.pop(i)
   Y_Train.pop(i)


# Making Testing Dataset
for i in range(len(X_Test)):
  p = i + len(Y_Train)
  try:
    temp = count_vector(X[p])
    temp[94] = float(entropy_calculator(X[p]))
    temp[95] = float(calculate_length(X[p]))
    Y_Test[i] = y_test[i]
    X_Test[i] = temp

  except ValueError:
    remIndex.append(i)

for i in remIndex: ##removing bad data
  X_Test.pop(i)
  Y_Test.pop(i)

ff.fit(X_Train, Y_Train, batch_size =256, epochs = 10)

out =ff.predict(X_Test)

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
total_val = len(out)
print()
for i in range(total_val):
  score = ""
  if out[i][0] < 0.5 :
    if(Y_Test[i] == 0):
      score = "Correct"
      false_neg += 1
    else:
      score = "Wrong"
      false_pos+= 1
    #print("Not-Mallicious", "Probability=", (1-pred_test[i])*100,"%", score)
  else:
    if(Y_Test[i] == 1):
      score = "Correct"
      true_pos += 1
    else:
      score = "Wrong"
      true_neg+= 1
    #print("Mallicious", "Probability=", pred_test[i]*100,"%", score)
countscore = false_pos + true_neg
print("Accuracy is", ((total_val - countscore)/total_val)*100, "%")
print("False-Positive: ", (false_pos/total_val)*100, "%")
print("False-Negative: ", (false_neg/total_val)*100, "%")
print("True-Positive: ", (true_pos/total_val)*100, "%")
print("True-Negative: ", (true_neg/total_val)*100, "%")