import os
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np

datas=[]
labellist=[]

for i in range(500):
	num = random.randrange(1,10)
	one_hot = np.zeros(shape=9)
	one_hot[num-1] = 1
	datas.append(one_hot)
	labellist.append(one_hot)
	
datas=np.array(datas)
	
log_dir = './logs'
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

inputs = keras.layers.Input(shape=(9,))
x=keras.layers.Dense(63,activation='relu')(inputs)
x=keras.layers.Dense(63,activation='relu')(x)
predictions=keras.layers.Dense(9,activation='softmax')(x)

model=keras.models.Model(inputs=inputs, outputs=predictions)


print("hello world!")
