#!/usr/bin/python3

# Machine Learning programm written using TensorFlow
# Data used to train the neural network come from a computer simulated 2D Ising
#  model, the purpose is to identify critical phase transitions using a trained
#  neural network, without feeding it with the order parameter.

import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt





def read_data(input_set, critical_temp):

    """Read data from file.

    Only argument is the path to the data file.
    
    File format:
    - odd lines contain magnetization and temperature separated by spaces
    - even lines contain spin configuration, single spin separated by spaces
    """

    magnetizations = []
    binary_temperatures = []
    real_temperatures = []
    configurations = []
    odd = True

    with open (input_set, 'r') as infile:
        for line in infile:
            if odd == True:
                magnetization, temperature = line.split()
                magnetizations.append(float(magnetization))
                temperature = float(temperature)
                real_temperatures.append(temperature)
                if temperature < critical_temp:
                    binary_temperatures.append(np.array([1,0]))
                else:
                    binary_temperatures.append(np.array([0,1]))
                odd = False

            else:
                configuration = np.fromstring(line, dtype=int, sep=' ')
                configurations.append(configuration)
                odd = True

    magnetizations = np.array(magnetizations)
    binary_temperatures = np.array(binary_temperatures)
    real_temperatures = np.array(real_temperatures)
    configurations = np.array(configurations)

    return magnetizations, binary_temperatures, real_temperatures, configurations



def build_model(data_shape, neurons_number):

    """Build neural network model. 
    """

    model = keras.Sequential([
        keras.layers.Dense(neurons_number, 
            activation=tf.sigmoid,
            kernel_initializer=keras.initializers.RandomNormal(stddev=1),  
            bias_initializer=keras.initializers.RandomNormal(stddev=1),
            kernel_regularizer=keras.regularizers.l2(0.003),
            input_shape=(data_shape,)),
 #       keras.layers.Dropout(0.2),
        keras.layers.Dense(2,
            activation=tf.nn.softmax,
            #kernel_initializer=tf.constant_initializer(np.array([[2, 1, -1], [-2, -2, 1]])),
            #bias_initializer=tf.constant_initializer(np.array([0, 0])))
            kernel_initializer=keras.initializers.RandomNormal(stddev=1),  
            bias_initializer=keras.initializers.RandomNormal(stddev=1))
        ])

    optimizer = tf.train.AdamOptimizer(0.001)

    model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'binary_crossentropy'])
    return model


square_temp = 2/np.log(1+np.sqrt(2))
triangular_temp = 4/np.log(3)
cubic_temp = 1/0.221654

test_temp = square_temp

train_set = sys.argv[1]
train_magns, train_bin_temps, train_real_temps, train_configs = read_data(train_set, square_temp)
test_set = sys.argv[2]
test_magns, test_bin_temps, test_real_temps, test_configs = read_data(test_set, test_temp)

neurons_number = 100
model = build_model(train_configs.shape[1], neurons_number)
model.summary()

#Calculate number of training set configurations to give to validation set (80%-20%)
val_perc = int(train_configs.shape[0]*20/100)

config_val = train_configs[:val_perc]
config_train_part = train_configs[val_perc:]

temp_val = train_bin_temps[:val_perc]
temp_train_part = train_bin_temps[val_perc:]

history = model.fit(config_train_part,
        temp_train_part,
        epochs=4,
        batch_size=128,
        validation_data=(config_val, temp_val),
        verbose=1)

# evaluate model using test dataset
results = model.evaluate(test_configs, test_bin_temps)
print("\nTest loss = " + str(results[0]) + "\nTest accuracy = " + str(results[1]))

# predict label on test dataset
predictions = model.predict(test_configs)

single_real_temps = np.round_(np.linspace(1, 5, 41), decimals=2)
predictions_t1 = []
predictions_t2 = []

# divide data for equal real temperatures 
for i in range(len(single_real_temps)):
    tmp_array = np.extract(test_real_temps==single_real_temps[i], predictions[:,0])
    predictions_t1.append(
            np.array([np.mean(tmp_array),
            np.std(tmp_array)/np.sqrt(len(tmp_array))]))
    tmp_array = np.extract(test_real_temps==single_real_temps[i], predictions[:,1])
    predictions_t2.append(
            np.array([np.mean(tmp_array), 
            np.std(tmp_array)/np.sqrt(len(tmp_array))]))

predictions_t1 = np.array(predictions_t1)
predictions_t2 = np.array(predictions_t2)

x = single_real_temps
y1 = predictions_t1[:,0]
y1_e = predictions_t1[:,1]
y2 = predictions_t2[:,0]
y2_e = predictions_t2[:,1]
plt.axvline(x=test_temp, marker='|', c='g', label='Trans temp')
plt.errorbar(x, y1, y1_e, c='b', marker='.', linewidth=2, label='No.1')
plt.errorbar(x, y2, y2_e, c='r', marker='.', linewidth=2, label='No.2')
plt.legend()
plt.show()

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

#y = np.matmul(test_configs, weights)+bias
#x = test_magns


#for i in range(0, neurons_number):
 #   plt.scatter(x, y[:,i], c=np.random.rand(3,1), marker='_')

#plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
binary_crossentropy = history.history['binary_crossentropy']
val_binary_crossentropy = history.history['val_binary_crossentropy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'g--', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, binary_crossentropy, 'r', label='Training crossentropy')
plt.plot(epochs, val_binary_crossentropy, 'r--', label='Validation crossentropy')
plt.xlabel('Epochs')
plt.ylabel('Binary_crossentropy')
plt.legend()

plt.show()
