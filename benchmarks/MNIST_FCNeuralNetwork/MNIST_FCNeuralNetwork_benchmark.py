#!/usr/bin/env python

import tensorflow as tf
import tensorflow.keras as keras
import time
import sys
import GPUtil
# from timeit import default_timer as timer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from functools import reduce
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from threading import Thread
from numba import cuda
import nvidia_smi
import numpy as np
from keras.utils.layer_utils import count_params

net_size_noneurons = []
network_size_noparam = []
gpu_load_train = []
gpu_mem_train = []
time_train = []
gpu_load_inf_in = []
gpu_mem_inf_in = []
time_inf_in = []
gpu_load_inf_out = []
gpu_mem_inf_out = []
time_inf_out = []

n_of_neurs = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,
              2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,
              60000,70000,80000,90000,100000,200000,300000,400000,500000]
batch_sizes = [50,200,400,600,800,1000,1200,1400,1800,2000]
# batch_sizes = [128]
# n_of_neurs = [10,20,30,1000,10000]

tf.keras.backend.clear_session()

"""
Say to GPU to use only the needed amount of memory
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

input_size = 28*28
output_size = 10
n_epochs = 1
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
    print("Matmul benchmark need 3 arguments:")
    print("- Device")
    print("- Shallow Layer dimension")
    print("- Batch size")
    sys.exit(1)

hidden_layer_list = []  # This list is read from the settings file

dev = sys.argv[1]
# hidden_layer_list.append(int(sys.argv[2]))
# batch_size = int(sys.argv[3])
batch_size = 256

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
        # print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

    def on_train_batch_begin(self, batch, logs={}):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_load.append(res.gpu)
        self.gpu_mem.append(res.memory)

    def on_train_batch_end(self, batch, logs={}):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_load.append(res.gpu)
        self.gpu_mem.append(res.memory)

class GPUstatistics_time(Thread):

    def __init__(self, delay):
        super(GPUstatistics_time, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        self.gpu_load = []
        self.gpu_mem = []
        while not self.stopped:
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_load.append(res.gpu)
            self.gpu_mem.append(res.memory)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


"""
Timing callback definition
"""


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_times = []
        self.epoch_times = []
        self.training_time = []
        self.training_time_start = time.time()

    def on_predict_begin(self, logs={}):
        self.batch_times = []
        self.epoch_times = []
        self.training_time = []
        self.training_time_start = time.time()

    def on_train_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.time()

    def on_train_batch_end(self, batch, logs={}):
        self.batch_times.append(time.time() - self.batch_time_start)

    def on_train_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_train_epoch_end(self, batch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, batch, logs={}):
        self.training_time.append(time.time() - self.training_time_start)
    
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


def data_loading(output):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Data preprocessing: I have to rescale and flatten all the images
    shape = (28, 28)
    shape_l = reduce(lambda a, b: a*b, shape)
    x_train = x_train.reshape((-1, shape_l)) / 255.
    x_test = x_test.reshape((-1, shape_l)) / 255.
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=output)
    y_test = to_categorical(y_test, num_classes=output)
    return (x_train, y_train), (x_test, y_test)


"""
Model definition method
"""


def model_def(hidden_layer, input, output):
    model = Sequential()
    for i in range(len(hidden_layer)+1):
        if i == 0:
            model.add(Dense(hidden_layer[i], activation='sigmoid',
                      input_shape=(input_size,)))
        elif i == len(hidden_layer):
            model.add(Dense(output_size, activation='softmax'))
        else:
            model.add(Dense(hidden_layer[i], activation='relu'))
    loss = keras.losses.CategoricalCrossentropy()
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.05)
    # metrics = ["accuracy"]
    # model.compile(loss=loss, optimizer=optim, metrics=metrics)
    model.compile(loss=loss, optimizer=optim)
    return model


def main(batch_size):
    tf.keras.backend.clear_session()

    (X_train, Y_train), (X_test, Y_test) = data_loading(output_size)
    nn = model_def(hidden_layer_list, input_size, output_size)
    # nn.summary()
    trainable_count = count_params(nn.trainable_weights)

    print('batch_size: {}  neurons: {}  NofParam: {}'
          .format(batch_size, n_of_neur, trainable_count) )

    """
    Training
    """

    time_callback = TimeHistory()
    # print("\nTraining...\n")
    # GPUstats = GPUstatistics()
    GPUstats = GPUstatistics_time(0.005)
    nn = model_def(hidden_layer_list, input_size, output_size)
    # monitor = Monitor(gpu_performance_sampling_time)     # GPU MONITOR

    nn.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size,
        #    callbacks=[time_callback, GPUstats],
        callbacks=[time_callback],
        #    validation_split=0.3,
           verbose=0)

    # monitor.stop()                                       # GPU MONITOR
    GPUstats.stop()
    gpu_loads_training = GPUstats.gpu_load
    gpu_mems_training = GPUstats.gpu_mem
    training_time = sum(time_callback.batch_times)
    time_per_sample = training_time/((len(X_train)//batch_size)
                                                      * batch_size)
    training_sample_per_second = 1./time_per_sample
    # print(gpu_mems)

    """
    Testing In-Sample
    """

    # print("\nTesting in-sample...\n")
    # monitor = Monitor(gpu_performance_sampling_time)     # GPU MONITOR

    time_callback = TimeHistory()
    # GPUstats = GPUstatistics()
    GPUstats = GPUstatistics_time(0.0001)
    pred = nn.predict(X_train, batch_size=batch_size,
                      callbacks=[time_callback]).argmax(1)

    # monitor.stop()                                       # GPU MONITOR
    GPUstats.stop()
    testing_time_insample = sum(time_callback.batch_times)
    gpu_loads_testin = GPUstats.gpu_load
    gpu_mems_testin = GPUstats.gpu_mem
    # accuracy_score(Y_train.argmax(1),
    #                 pred, normalize=True)
    time_per_sample = testing_time_insample/len(X_train)
    test_insample_sample_per_second = 1./time_per_sample

    """
    Testing Out-of-Sample
    """

    # print("\nTesting out-of-sample...\n")
    # monitor = Monitor(gpu_performance_sampling_time)     # GPU MONITOR

    time_callback = TimeHistory()
    # GPUstats = GPUstatistics()
    GPUstats = GPUstatistics_time(0.005)
    pred = nn.predict(X_test, batch_size=batch_size,
                      callbacks=[time_callback]).argmax(1)

    # monitor.stop()                                       # GPU MONITOR
    GPUstats.stop()
    testing_time_outofsample = sum(time_callback.batch_times)
    gpu_loads_testout = GPUstats.gpu_load
    gpu_mems_testout = GPUstats.gpu_mem
    accuracy = accuracy_score(Y_test.argmax(1),
                   pred, normalize=True)
    time_per_sample = testing_time_outofsample/len(X_test)
    test_outofsample_sample_per_second = 1./time_per_sample

    net_size_noneurons[len(net_size_noneurons)-1].append(n_of_neur)
    network_size_noparam[len(network_size_noparam)-1].append(trainable_count)
    gpu_load_train[len(gpu_load_train)-1].append(
        [np.asarray(gpu_loads_training).mean(),np.asarray(gpu_loads_training).min(),np.asarray(gpu_loads_training).max()])
    
    gpu_mem_train[len(gpu_mem_train)-1].append(
        [np.asarray(gpu_mems_training).mean(),np.asarray(gpu_mems_training).min(),np.asarray(gpu_mems_training).max()])
    
    time_train[len(time_train)-1].append(training_time)
    
    gpu_load_inf_in[len(gpu_load_inf_in)-1].append(
        [np.asarray(gpu_loads_testin).mean(),np.asarray(gpu_loads_testin).min(),np.asarray(gpu_loads_testin).max()])
    
    gpu_mem_inf_in[len(gpu_mem_inf_in)-1].append(
        [np.asarray(gpu_mems_testin).mean(),np.asarray(gpu_mems_testin).min(),np.asarray(gpu_mems_testin).max()])
    
    time_inf_in[len(time_inf_in)-1].append(testing_time_insample)
    
    gpu_load_inf_out[len(gpu_load_inf_out)-1].append(
        [np.asarray(gpu_loads_testout).mean(),np.asarray(gpu_loads_testout).min(),np.asarray(gpu_loads_testout).max()])
    
    gpu_mem_inf_out[len(gpu_mem_inf_out)-1].append(
        [np.asarray(gpu_mems_testout).mean(),np.asarray(gpu_mems_testout).min(),np.asarray(gpu_mems_testout).max()])
    
    time_inf_out[len(time_inf_out)-1].append(testing_time_outofsample)

    """
    free gpu memory
    """

#     device = cuda.get_current_device()
#     device.reset()
    return 0


if __name__ == "__main__":
    with tf.device(dev):
        for batch in batch_sizes:
            net_size_noneurons.append([])
            network_size_noparam.append([])
            gpu_load_train.append([])
            gpu_mem_train.append([])
            time_train.append([])
            gpu_load_inf_in.append([])
            gpu_mem_inf_in.append([])
            time_inf_in.append([])
            gpu_load_inf_out.append([])
            gpu_mem_inf_out.append([])
            time_inf_out.append([])
            for n_of_neur in n_of_neurs:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                hidden_layer_list = [n_of_neur]  # This list is read from the settings file
                main(batch)
    print(np.asarray(net_size_noneurons).shape, np.asarray(network_size_noparam).shape,
          np.asarray(gpu_load_train).shape, np.asarray(gpu_mem_train).shape, np.asarray(time_train).shape,
          np.asarray(gpu_load_inf_in).shape, np.asarray(gpu_mem_inf_in).shape, np.asarray(time_inf_in).shape,
          np.asarray(gpu_load_inf_out).shape, np.asarray(gpu_mem_inf_out).shape, np.asarray(time_inf_out).shape)
    np.savez('net_size_noneurons_MON',net_size_noneurons)
    np.savez('network_size_noparam_MON',network_size_noparam)
    np.savez('gpu_load_train_MON',gpu_load_train)
    np.savez('gpu_mem_train_MON',gpu_mem_train)
    np.savez('time_train_MON',time_train)
    np.savez('gpu_load_inf_in_MON',gpu_load_inf_in)
    np.savez('gpu_mem_inf_in_MON',gpu_mem_inf_in)
    np.savez('time_inf_in_MON',time_inf_in)
    np.savez('gpu_load_inf_out_MON',gpu_load_inf_out)
    np.savez('gpu_mem_inf_out_MON',gpu_mem_inf_out)
    np.savez('time_inf_out_MON',time_inf_out)

