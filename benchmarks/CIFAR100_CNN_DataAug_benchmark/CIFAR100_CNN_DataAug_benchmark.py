#!/usr/bin/env python

import tensorflow as tf
import tensorflow.keras as keras
import time
import sys
import GPUtil
# from timeit import default_timer as timer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import cifar100
from sklearn.metrics import accuracy_score
from threading import Thread
from numba import cuda
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import nvidia_smi
import numpy as np

tf.keras.backend.clear_session()


"""
Say to GPU to use onky the needed amount of memory
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

"""
Model parameters
"""
input_shape = (32, 32, 3)
output_size = 100
batch_size = 500
n_epochs = 2
learning_rate = 0.01
gpu_performance_sampling_time = 1
gpu_performance_sampling_time_INFERENCE = 0.1

"""
READ FROM COMMAND LINE DEVICE, NETWORK SIZE AND BATCH SIZE
CHECK GPU AVAILABILITY
"""
if tf.config.list_physical_devices('GPU') == 0:
    print("GPU unavailable :(")
    sys.exit(0)

"""
READ ARGS FROM COMMAND LINE
Raise error if correct arguments aren't given
"""
if len(sys.argv) != 4:
    print("CIFAR100_CNN benchmark need 3 arguments:")
    print("- Device")
    print("- Number of Epochs")
    print("- Data Augmentation")
    sys.exit(1)

dev = sys.argv[1]
n_epochs = int(sys.argv[2])
aug_data = bool(sys.argv[3])

"""
SET DEVICE
"""
if dev == 'cpu':
    d = '/cpu:0'
elif dev == 'gpu':
    if tf.config.list_physical_devices('GPU') == 0:
        print("GPU unavailable :(")
        sys.exit(0)
    d = '/device:GPU:0'


"""
GPU USAGE MONITOR
"""


class Monitor(Thread):

    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


"""
Callback to get GPU usage statistics
"""


class GPUstatistics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.gpu_load = []
        self.gpu_mem = []

    def on_predict_begin(self, logs={}):
        self.gpu_load = []
        self.gpu_mem = []

    def on_predict_batch_begin(self, batch, logs={}):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_load.append(res.gpu)
        self.gpu_mem.append(res.memory)

    def on_predict_batch_end(self, batch, logs={}):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_load.append(res.gpu)
        self.gpu_mem.append(res.memory)

    def on_train_batch_begin(self, batch, logs={}):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_load.append(res.gpu)
        self.gpu_mem.append(res.memory)

    def on_train_batch_end(self, batch, logs={}):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_load.append(res.gpu)
        self.gpu_mem.append(res.memory)


"""
Timing callback definition
"""


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_times = []
        self.epoch_times = []
        self.training_time = []
        self.training_time_start = time.time()

    def on_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        self.batch_times.append(time.time() - self.batch_time_start)

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, batch, logs={}):
        self.training_time.append(time.time() - self.training_time_start)

    def on_predict_begin(self, logs={}):
        self.batch_times = []
        self.epoch_times = []
        self.training_time = []
        self.training_time_start = time.time()

    def on_predict_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.time()

    def on_predict_batch_end(self, batch, logs={}):
        self.batch_times.append(time.time() - self.batch_time_start)

    def on_predict_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_predict_epoch_end(self, batch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_time_start)

    def on_predict_end(self, batch, logs={}):
        self.training_time.append(time.time() - self.training_time_start)


"""
Data loading method
"""


def data_loading():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # Data preprocessing: I have to rescale the images
    x_train, x_test = x_train/255., x_test/255.
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=output_size)
    y_test = to_categorical(y_test, num_classes=output_size)
    return (x_train, y_train), (x_test, y_test)


"""
Model definition method
"""


def model_def():
    model = Sequential()
    model.add(Conv2D(64, (4, 4), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(128, (4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(256, (4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    loss = keras.losses.CategoricalCrossentropy()
    optim = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.05)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model


def main():
    """
    Augmented Data Generator
    """
    datagen = ImageDataGenerator(
                                # set input mean to 0 over the dataset
                                featurewise_center=True,
                                # set each sample mean to 0
                                samplewise_center=False,
                                # divide inputs by std of the dataset
                                featurewise_std_normalization=True,
                                # divide each input by its std
                                samplewise_std_normalization=False,
                                # apply ZCA whitening
                                zca_whitening=False,
                                # randomly rotate images in the range (degrees, 0 to 180)
                                rotation_range=0,
                                # randomly shift images horizontally
                                # (fraction of total width)
                                width_shift_range=0.1,
                                # randomly shift images vertically
                                # (fraction of total height)
                                height_shift_range=0.1,
                                # randomly flip images
                                horizontal_flip=True,
                                # randomly flip images
                                vertical_flip=False)

    (x_train, y_train), (X_test, Y_test) = data_loading()
    datagen.fit(x_train)
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1)

    """
    Training
    """

    print("\nTRAINING...\n")
    time_callback = TimeHistory()
    nn = model_def()
    GPUstats = GPUstatistics()
    # monitor = Monitor(gpu_performance_sampling_time)     # GPU MONITOR
    # begin = timer()  # Duration of the whole fit() method run

    if aug_data:
        nn.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
               epochs=n_epochs,
               batch_size=batch_size,
               steps_per_epoch=len(X_train) // batch_size,
               validation_data=datagen.flow(X_val, Y_val, batch_size=batch_size),
               verbose=0,
               callbacks=[time_callback, GPUstats])
    else:
        nn.fit(X_train, Y_train,
               epochs=n_epochs,
               batch_size=batch_size,
               validation_data=(X_val, Y_val),
               verbose=0,
               callbacks=[time_callback, GPUstats])

    # training_time = timer() - begin  # Duration of the whole fit() method run
    gpu_loads = GPUstats.gpu_load
    gpu_mems = GPUstats.gpu_mem
    # monitor.stop()                                       # GPU MONITOR
    training_time_sum_over_batches = sum(time_callback.batch_times)
    time_per_sample = training_time_sum_over_batches/((len(X_train)//batch_size)
                                                      * batch_size)
    sample_per_second = 1./time_per_sample

    """
    Testing In-Sample
    """

    print("\nTESTING in-sample...\n")
    # monitor = Monitor(gpu_performance_sampling_time)     # GPU MONITOR
    GPUstats = GPUstatistics()
    # begin = timer()  # Inference time on training set
    pred = nn.predict(X_train, batch_size=batch_size,
                      callbacks=[time_callback, GPUstats]).argmax(1)
    # testing_time_insample = timer() - begin  # Inference time on training set
    # monitor.stop()                                       # GPU MONITOR
    testing_time_insample = sum(time_callback.batch_times)
    gpu_loads_testin = GPUstats.gpu_load
    gpu_mems_testin = GPUstats.gpu_mem
    accuracy = accuracy_score(Y_train.argmax(1),
                              pred, normalize=False)/len(X_train)
    time_per_sample_test_insample = testing_time_insample/len(X_train)
    sample_per_second_test_insample = 1./time_per_sample_test_insample

    """
    Testing Out-of-Sample
    """

    print("\nTESTING out-of-sample...\n")
    # monitor = Monitor(gpu_performance_sampling_time)     # GPU MONITOR
    GPUstats = GPUstatistics()
    # begin = timer()  # Inference time on training set
    pred = nn.predict(X_test, batch_size=batch_size,
                      callbacks=[time_callback, GPUstats]).argmax(1)
    # testing_time_outofsample = timer() - begin  # Inference time on training set
    # monitor.stop()                                      # GPU MONITOR
    testing_time_outofsample = sum(time_callback.batch_times)
    gpu_loads_testout = GPUstats.gpu_load
    gpu_mems_testout = GPUstats.gpu_mem
    accuracy = accuracy_score(Y_test.argmax(1),
                              pred, normalize=False)/len(X_test)
    time_per_sample_test_outofsample = testing_time_outofsample/len(X_test)
    sample_per_second_test_outofsample = 1./time_per_sample_test_outofsample

    """
    free gpu memory
    """

    device = cuda.get_current_device()
    device.reset()

    print("\nAverage GPU usage during training: %s\n" % np.asarray(gpu_loads).mean())
    print("\nAverage GPU memory usage during training: %s\n"
          % np.asarray(gpu_mems).mean())

    print("\nAverage GPU usage during inference in-sample: %s\n"
          % np.asarray(gpu_loads_testin).mean())
    print("\nAverage GPU memory usage during inference in-sample: %s\n"
          % np.asarray(gpu_mems_testin).mean())

    print("\nAverage GPU usage during inference out-of-sample: %s\n"
          % np.asarray(gpu_loads_testout).mean())
    print("\nAverage GPU memory usage during inference out-of-sample: %s\n"
          % np.asarray(gpu_mems_testout).mean())

    print("\nAccurcay over test set: %s\n" % accuracy)

    print("\nTraining the model took %s seconds\n"
          % training_time_sum_over_batches)

    print("\nSample processing time during training: %s sample/seconds\n"
          % sample_per_second)

    print("\nTesting the model in-sample took %s seconds\n" % testing_time_insample)

    print("\nSample processing time during inference in-sample: %s sample/seconds\n"
          % sample_per_second_test_insample)

    print("\nTesting the model out-of-sample took %s seconds\n"
          % testing_time_outofsample)

    print("\nSample processing time during inference out-of-sample: %s sample/seconds\n"
          % sample_per_second_test_outofsample)

    return 0


if __name__ == "__main__":
    with tf.device(dev):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        main()
