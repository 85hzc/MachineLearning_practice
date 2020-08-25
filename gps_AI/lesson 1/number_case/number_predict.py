import  tensorflow as tf
from tensorflow import keras
import random
import numpy as np
datas=[]
labellist=[]
for i in range(500):
    num=random.randrange(1,10)
    one_hot=np.zeros(shape=9)
    one_hot[num-1]=1
    datas.append(one_hot)
    labellist.append(one_hot)

datas=np.array(datas)
labellist=np.array(labellist)


log_dir = './logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
inputs = keras.layers.Input(shape=(9,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(9, activation='softmax')(x)
model = keras.models.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='sgd',loss=tf.keras.losses.CategoricalCrossentropy() )
print("================开始训练======================")
model.fit(datas, labellist,epochs=100,
              verbose=1,callbacks=[tensorboard_callback])  # 开始训练
print("================开始预测======================")
for i in range(10):
    num=random.randrange(0, 499)
    real_num=np.argmax(datas[num])+1
    input=datas[num]
    predict=np.argmax(model.predict(np.expand_dims(input, axis=0))) + 1
    print("predict num is %d ,real num is %d"% (predict,real_num))