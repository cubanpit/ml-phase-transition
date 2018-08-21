#!/usr/bin/python3

import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def read_data(input_set):

    magnetizations = []
    temperatures = []
    configurations = []
    odd = True

    with open (input_set, 'r') as infile:
        for line in infile:
            if odd == True:
                magnetization, temperature = line.split()
                magnetizations.append(float(magnetization))
                if float(temperature) < 2/np.log(1+np.sqrt(2)):
                    temperatures.append(np.array([1,0]))
                else:
                    temperatures.append(np.array([0,1]))
                odd = False

            else:
                configuration = np.fromstring(line, dtype=int, sep=' ')
                configurations.append(configuration)
                odd = True

    magnetizations = np.array(magnetizations)
    temperatures = np.array(temperatures)
    configurations = np.array(configurations)

    return magnetizations, temperatures, configurations

def build_model(data_shape):
    model = keras.Sequential([
        keras.layers.Dense(5, 
            activation=tf.sigmoid,
            kernel_initializer=keras.initializers.RandomNormal(stddev=1),  
            bias_initializer=keras.initializers.RandomNormal(stddev=1),
            kernel_regularizer=keras.regularizers.l2(0.1),
            input_shape=(data_shape,)),
        keras.layers.Dense(2,
            activation=tf.nn.softmax)
#            kernel_initializer=tf.constant_initializer(np.array([[2, 1, -1], [-2, -2, 1]])),
#            bias_initializer=tf.constant_initializer(np.array([0, 0])))
        ])

    optimizer = tf.train.AdamOptimizer()

    model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    return model


train_set = sys.argv[1]
train_magns, train_temps, train_configs = read_data(train_set)
test_set = sys.argv[2]
test_magns, test_temps, test_configs = read_data(test_set)

model = build_model(train_configs.shape[1])
model.summary()

config_val = train_configs[:10000]
config_train_part = train_configs[10000:]

temp_val = train_temps[:10000]
temp_train_part = train_temps[10000:]

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

y = np.matmul(test_configs, weights)+bias
x = test_magns

plt.scatter(x, y[:,0], c='b', marker='*', label='No.1')
plt.scatter(x, y[:,1], c='y', marker='1', label='No.2')
plt.scatter(x, y[:,2], c='g', marker='+', label='No.3')
plt.scatter(x, y[:,3], c='r', marker='_', label='No.4')
plt.scatter(x, y[:,4], c='c', marker='+', label='No.5')
plt.legend()
plt.show()

history = model.fit(config_train_part,
        temp_train_part,
        epochs=20,
        batch_size=100,
        validation_data=(config_val, temp_val),
        verbose=1)

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

y = np.matmul(test_configs, weights)+bias
x = test_magns


#for i in range(0, 99):
#    plt.scatter(x, y[:,i])

plt.scatter(x, y[:,0], c='b', marker='*', label='No.1')
plt.scatter(x, y[:,1], c='y', marker='1', label='No.2')
plt.scatter(x, y[:,2], c='g', marker='+', label='No.3')
plt.scatter(x, y[:,3], c='r', marker='_', label='No.4')
plt.scatter(x, y[:,4], c='c', marker='+', label='No.5')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#plt.plot(epochs, loss, 'ro', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')

history_dict = history.history
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy or loss')
plt.legend()

plt.show()

