#!/usr/bin/python3

# File: cv_ising_learning.py

# Machine Learning programm written using TensorFlow
# Data used to train the neural network come from a computer simulated 2D Ising
#  model, the purpose is to identify critical phase transitions using a trained
#  neural network, without feeding it with the order parameter.
# WARNING: this model does not perform well, a simpler approach is with a
#  fully connected neural network (see file ising_learning.py). This file could
#  be messy and contain leftovers from testing work.

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split, KFold

parser = argparse.ArgumentParser()
parser.add_argument(
        "-tr", "--training_set",
        help="Training set file",
        required=False)
parser.add_argument(
        "test_set", help="Test set file")
parser.add_argument(
        "-lt", "--lattice_type",
        help="Test set lattice type: square (sq), triangular (tr),\
        honeycomb (hc), cubic (cb)", required=True)
parser.add_argument(
        "-nn", "--neurons_number",
        help="Neuron number for FNN",
        required=False, type=int, default=100)
parser.add_argument(
        "-lm", "--load_models",
        help="Files '.h5' containing previously trained model(s)",
        required=False, nargs='+')
parser.add_argument(
        "-sm", "--save_model",
        help="File to save trained model(s), if there are multiple models\
                they will be saved in numbered files with this common prefix,\
                the '.h5' extension is added to every filename",
        required=False)
parser.add_argument(
        "-db", "--debug",
        help="Enable every plot and every output, useful to follow \
                network training and performance.",
        required=False, action='store_true')
args = parser.parse_args()


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

    with open(input_set, 'r') as infile:
        for line in infile:
            if odd is True:
                infos = line.split()
                if len(infos) == 2:
                    magnetization, temperature = infos[:]
                elif len(infos) == 1:
                    temperature = infos[0]
                    magnetization = 0.0
                else:
                    raise RuntimeError(
                            "Wrong number of information on the same line.\n"
                            "Expected informations: (magnetization) "
                            "temperature")
                temperature = float(temperature)
                real_temperatures.append(temperature)
                if temperature < critical_temp:
                    binary_temperatures.append(np.array([1, 0]))
                else:
                    binary_temperatures.append(np.array([0, 1]))
                odd = False

            else:
                configuration = np.fromstring(line, dtype=np.int8, sep=' ')
                configurations.append(configuration)
                odd = True

    magnetizations = np.array(magnetizations).astype(np.float32)
    binary_temperatures = np.array(binary_temperatures).astype(np.uint8)
    real_temperatures = np.array(real_temperatures).astype(np.float32)
    configurations = np.array(configurations).astype(np.int8)

    return magnetizations, binary_temperatures, real_temperatures, configurations


def critical_temp(input_lattice):
    """Returns critical temperature for different lattice.
    """

    square_temp = 2/np.log(1+np.sqrt(2))
    triangular_temp = 4/np.log(3)
    cubic_temp = 1/0.221654
    honeycomb_temp = 1/0.658478

    if input_lattice == "sq":
        test_temp = square_temp
    elif input_lattice == "tr":
        test_temp = triangular_temp
    elif input_lattice == "hc":
        test_temp = honeycomb_temp
    elif input_lattice == "cb":
        test_temp = cubic_temp
    else:
        raise SyntaxError("Use sq for square, tr for triangular and cb for cubic")

    return test_temp


def unique_elements(complete_array):
    """Returns a list of different elements in an array.
    """

    uniques = []
    for elem in complete_array:
        if elem not in uniques:
            uniques.append(elem)

    uniques = np.array(uniques)
    uniques.sort()

    return uniques


def unison_shuffled_copies(a, b):
    """Shuffle two arrays with corresponding elements.
        High memory usage, makes entire copy of arrays.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def line_eq(p1, p2):
    """Returns coefficient of a line equation given two points.
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection_pt(L1, L2):
    """Returns intersection point coordinates given two lines.
    """
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def build_model(data_shape, neurons_number):
    """Build neural network model with given data shape and neurons number.
    """

    model = keras.Sequential([
        keras.layers.Dense(
            neurons_number,
            activation=tf.sigmoid,
            kernel_initializer=keras.initializers.RandomNormal(stddev=1),
            bias_initializer=keras.initializers.RandomNormal(stddev=1),
            kernel_regularizer=keras.regularizers.l2(0.001),
            input_shape=(data_shape,)),
    #   keras.layers.Dropout(0.30),
        keras.layers.Dense(
            2,
            activation=tf.nn.softmax,
            # kernel_initializer=tf.constant_initializer(np.array([[2, 1, -1], [-2, -2, 1]])),
            # bias_initializer=tf.constant_initializer(np.array([0, 0])))
            kernel_initializer=keras.initializers.RandomNormal(stddev=1),
            bias_initializer=keras.initializers.RandomNormal(stddev=1))
        ])

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'binary_crossentropy']
            )

    return model


def train_model(model, training_inputs, training_labels):
    """Train given model with given data.

    Returns history of keras fit function.
    """
    # define callback to stop when accuracy is stable
    earlystop = keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.0001,
            patience=4, verbose=1, mode='auto')
    callbacks_list = [earlystop]

    return model.fit(
            training_inputs, training_labels,
            validation_split=0.2, epochs=500,
            callbacks=callbacks_list, batch_size=100,
            shuffle=True, verbose=0)


#
#   MAIN
#

if args.training_set is not None:
    train = True

    if args.save_model is None:
        save = False
    else:
        save = True

    if args.load_models is not None:
        raise SyntaxError("You can not load a model and train a new one, choose\
                between the two options.")
else:
    if args.load_models is None:
        raise SyntaxError("You must have a training set or \
                            a previously trained model.")
    else:
        train = False
        save = False

        if args.save_model is not None:
            raise SyntaxError("You can not load a saved model and save it, it\
                    does not make any sense.")

# if there is a training set load it, otherwise load the trained model
models = []
if train:
    print("Training new model(s) using as training set:", args.training_set)
    train_set = args.training_set
    train_magns, train_bin_temps, train_real_temps, train_configs \
            = read_data(train_set, critical_temp("sq"))
    model = build_model(train_configs.shape[1], args.neurons_number)

    folds = list(KFold(n_splits=10, shuffle=True, random_state=1).split(train_configs, train_bin_temps))

    # define callback to stop when accuracy is stable
    earlystop = keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.0001,
            patience=6, verbose=1, mode='auto')

    callbacks_list = [earlystop]

    for j, (train_idx, val_idx) in enumerate(folds):

        print ('\nFold', j)
        config_train = train_configs[train_idx]
        config_val = train_configs[val_idx]

        temp_train = train_bin_temps[train_idx]
        temp_val = train_bin_temps[val_idx]

        # fit model on training data
        history = model.fit(
                        config_train,
                        temp_train,
                        epochs=30,
                        callbacks=callbacks_list,
                        batch_size=100,
                        validation_data=(config_val, temp_val),
                        verbose=1)

        if train and not args.no_plot:
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

            plt.plot(
                    epochs,
                    binary_crossentropy,
                    'r',
                    label='Training crossentropy')
            plt.plot(
                    epochs,
                    val_binary_crossentropy,
                    'r--',
                    label='Validation crossentropy')
            plt.xlabel('Epochs')
            plt.ylabel('Binary_crossentropy')
            plt.legend()

            plt.show()
    if save:
        print("Saving trained model to:", args.save_model)
        model.save(args.save_model)
else:
    print("Loading trained model(s) from:", args.load_models, "\n")
    for mf in args.load_models:
        models.append(keras.models.load_model(mf))

# print summary of first model, as reference
models[0].summary()
n_models = len(models)

# set test critical temperature based on lattice type
test_temp = critical_temp(args.lattice_type)

# load test set
test_set = args.test_set
test_magns, test_bin_temps, test_real_temps, test_configs \
        = read_data(test_set, test_temp)

# split test set in n_split sets, to compute statistical accuracy
n_split = 10
n_elem = int(len(test_bin_temps) / n_split) * n_split
many_test_bin_t = np.split(test_bin_temps[:n_elem], n_split)
many_test_real_t = np.split(test_real_temps[:n_elem], n_split)
many_test_configs = np.split(test_configs[:n_elem], n_split)
tc_predictions = []

for m in range(n_models):
    print("\nEvaluating model", m, ". . .")
    accuracies = []
    losses = []
    n_miss = 0
    miss = False
    for t in range(n_split):
        # evaluate model using test dataset
        results = models[m].evaluate(
                                     many_test_configs[t],
                                     many_test_bin_t[t],
                                     verbose=0)
        accuracies.append(results[1])
        losses.append(results[0])

        # predict label on test dataset
        predictions = models[m].predict(many_test_configs[t])

        # get a list of every temperature in complete test set
        single_real_temps = unique_elements(many_test_real_t[t])
        predictions_t1 = []
        predictions_t2 = []

        # divide data for equal real temperatures
        for temp in single_real_temps:
            tmp_array = \
                np.extract(many_test_real_t[t] == temp, predictions[:, 0])
            predictions_t1.append(
                    np.array([np.mean(tmp_array),
                              np.std(tmp_array) / np.sqrt(len(tmp_array) - 1)]))
            tmp_array = \
                np.extract(many_test_real_t[t] == temp, predictions[:, 1])
            predictions_t2.append(
                    np.array([np.mean(tmp_array),
                              np.std(tmp_array) / np.sqrt(len(tmp_array) - 1)]))

        predictions_t1 = np.array(predictions_t1)
        predictions_t2 = np.array(predictions_t2)
        xt = single_real_temps
        y1 = predictions_t1[:, 0]
        y2 = predictions_t2[:, 0]

        if args.debug:
            y1_e = predictions_t1[:, 1]
            y2_e = predictions_t2[:, 1]
            plt.axvline(x=test_temp, marker='|', c='g', label='Critical temperature')
            plt.errorbar(xt, y1, y1_e, c='b', marker='.', linewidth=2, label='No.1')
            plt.errorbar(xt, y2, y2_e, c='r', marker='.', linewidth=2, label='No.2')
            plt.legend()
            plt.show()

        # find first element greater than critical temp
        index_tc = next(
                        x[0] for x in enumerate(single_real_temps)
                        if x[1] > test_temp)

        # compute intersection of two lines passing for two given points each
        line1 = line_eq(
                [xt[index_tc-1], y1[index_tc-1]],
                [xt[index_tc], y1[index_tc]])
        line2 = line_eq(
                [xt[index_tc-1], y2[index_tc-1]],
                [xt[index_tc], y2[index_tc]])
        inters_point = intersection_pt(line1, line2)

        # if successful add it to the predictions array
        if not inters_point:
            miss = True
        elif inters_point[0] > xt[index_tc] or inters_point[0] < xt[index_tc-1]:
            miss = True
        else:
            tc_predictions.append(inters_point[0])
            miss = False

        if miss:
            n_miss += 1
            orig_index_tc = index_tc

            for i in range(-5, 5):
                if i == 0:
                    continue
                index_tc = orig_index_tc + i

                # compute intersection of two lines passing for two given points each
                line1 = line_eq(
                        [xt[index_tc-1], y1[index_tc-1]],
                        [xt[index_tc], y1[index_tc]])
                line2 = line_eq(
                        [xt[index_tc-1], y2[index_tc-1]],
                        [xt[index_tc], y2[index_tc]])
                inters_point = intersection_pt(line1, line2)

                # if successful add it to the predictions array
                if inters_point and inters_point[0] < xt[index_tc] \
                        and inters_point[0] > xt[index_tc-1]:
                    tc_predictions.append(inters_point[0])
                    miss = False
                    break

            if miss:
                print("No intersection near critical point!")

    print(
          "Average accuracy =",
          np.round(np.mean(accuracies), decimals=4),
          "+-",
          np.round(np.std(accuracies)/np.sqrt(len(accuracies) - 1), decimals=5)
          )
    print(
          "Average loss =",
          np.round(np.mean(losses), decimals=4),
          "+-",
          np.round(np.std(losses)/np.sqrt(len(losses) - 1), decimals=5)
          )
    print("Number of missed temperatures =", n_miss)

# compute mean and stdev
print("\nTotal number of elements =", len(tc_predictions))
if len(tc_predictions) > 1:
    tc_predictions = np.array(tc_predictions)
    tc_mean = np.round(np.mean(tc_predictions), decimals=4)
    tc_stdev = \
        np.round(np.std(tc_predictions)/np.sqrt(len(tc_predictions) - 1),
                 decimals=5)
    print("Predicted critical temperature: mean =", tc_mean, "+-", tc_stdev)
    print("Theoretical critical temperature =", np.round(test_temp, decimals=4))
else:
    print("There are no useful data,\
            impossible to compute critical temperature")

#y1_e = predictions_t1[:, 1]
#y2_e = predictions_t2[:, 1]
#plt.axvline(x=test_temp, marker='|', c='g', label='Critical temperature')
#plt.errorbar(xt, y1, y1_e, c='b', marker='.', linewidth=2, label='No.1')
#plt.errorbar(xt, y2, y2_e, c='r', marker='.', linewidth=2, label='No.2')
#plt.legend()
#plt.show()
#
#weights = model.layers[0].get_weights()[0]
#bias = model.layers[0].get_weights()[1]
#
#y = np.matmul(test_configs, weights)+bias
#x = test_magns
#
#for i in range(0, neurons_number):
#    plt.scatter(x, y[:,i], c=np.random.rand(3,1), marker='_')
#
#plt.show()

if train and not args.no_plot:
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

    plt.plot(
            epochs,
            binary_crossentropy,
            'r',
            label='Training crossentropy')
    plt.plot(
            epochs,
            val_binary_crossentropy,
            'r--',
            label='Validation crossentropy')
    plt.xlabel('Epochs')
    plt.ylabel('Binary_crossentropy')
    plt.legend()

    plt.show()

# Copyright 2018 Pietro F. Fontana <pietrofrancesco.fontana@studenti.unimi.it>
#                Martina Crippa    <martina.crippa2@studenti.unimi.it>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
