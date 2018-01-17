
# coding: utf-8

# In[1]:


import keras


# In[2]:


import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Permute
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, Lambda, Conv2D, Reshape

# 數學函式庫
import math

#讀資料用
import os, sys, csv
import pandas as pd
import pickle

import numpy as np
import tensorflow as tf


# In[3]:


Data = pd.read_excel("/Users/hsienhaochen/Documents/凱利公式+DeepLearning/TW93_train.xlsx")#Taiwan100 Stock Price Data


# In[4]:


Data_array = Data.values
print(Data_array[240])


# In[5]:


company = 93
days = 2000
column = 11

X_unit = 240
Y_unit = 10

num_of_data = 9


# In[6]:


# (天數, 公司數, 數據量)
Data_array = Data_array.reshape((company, days, column))

#拿掉日期＋公司名稱
Data_array = Data_array[:, :, 2:]


# In[7]:


Data_array.shape


# In[8]:


# each_data = all_data[:, 0:240, :]
# for i in range(1, 1760):
#     each_data = np.concatenate((each_data, all_data[:, i:i+240, :]), axis = 1)


# In[9]:


training_data = np.zeros(shape=(days-X_unit-Y_unit, company, X_unit, column-2))
training_data.shape


# In[10]:


for i in range(training_data.shape[1]):
    training_data[i, :, :, :] = Data_array[:, i:i+X_unit, :]


# In[11]:


training_data.shape


# In[12]:


X_train = training_data
X_train.shape


# In[13]:


days = X_train.shape[0] 
channel = 1


# In[14]:


X_train = X_train.reshape(days*company, X_unit, num_of_data, channel)


# In[15]:


X_train.shape


# In[16]:


num_classes = 4


# In[17]:


Y_train = np.random.randint(0, 3, (X_train.shape[0]))


# In[18]:


Y_train.shape


# In[19]:


Y_train


# In[20]:


Y_train = keras.utils.to_categorical(Y_train, num_classes=num_classes)


# In[21]:


Y_train.shape


# In[22]:


Y_train


# In[23]:


input_shape = X_train.shape[1:]
input_shape


# In[24]:


#Input
model_input = Input(shape = input_shape)

#隨便弄一個model
conv_1 = Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation = "relu")(model_input)
conv_2 = Conv2D(filters = 128, kernel_size = (3, 3), padding='same', activation = "relu")(conv_1)
conv_3 = Conv2D(filters = 256, kernel_size = (3, 3), padding='same', activation = "relu")(conv_2)

#拉直
flatten = Flatten()(conv_3)

#soft landing
soft_landing = Dense(2048, activation="relu")(flatten)

#動作有4種，所以最後輸出是4維
action = Dense(num_classes, activation="softmax")(soft_landing)


#把整個model包起來
model = Model(model_input, action)

#summary
model.summary()


# In[25]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=0.00025),
              metrics=['accuracy'])


# In[26]:


epochs = int(input("epochs?"))


# In[ ]:


model.fit(X_train, Y_train,
          epochs=epochs,
          verbose=1)

