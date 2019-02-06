#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz;
from sklearn import preprocessing

import numpy as np
import pandas as pd


# In[ ]:


print("\n\n--------------------------------------------------------\n\n")
print("\t\tFull F.E. Dataset")
print("\n\n--------------------------------------------------------\n\n")


# In[ ]:


train = pd.read_csv("../../data/train_FE.csv")
train = train[train['maxPlace'] > 1]


# In[ ]:


target = "winPlacePerc"
features = list(train.columns)
features.remove("Id")
features.remove("matchId")
features.remove("groupId")
features.remove("matchType")

y = np.array(train[target])
features.remove(target)
x = train[features]


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.1, random_state=0)


# In[ ]:


x_train = np.array(x_train)
x_val = np.array(x_val)


# In[ ]:


scale = preprocessing.MinMaxScaler(copy=False).fit(x_train)
x_train_trans = scale.transform(x_train)
x_val_trans = scale.transform(x_val)


# In[ ]:


model = Sequential()
model.add(Dense(x_train_trans.shape[1], input_dim=x_train_trans.shape[1], activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))

model.add(Dense(1, activation='linear'))


# In[ ]:


model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()


# In[ ]:


checkpoint_name = '../nn_weights/full_fe/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[ ]:


epochs = 100
batch_size = 1024


# In[ ]:


model.fit(x=x_train_trans, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(x_val_trans, y_val))


# In[ ]:


# training mae: 
# validation mae: 


# In[ ]:




