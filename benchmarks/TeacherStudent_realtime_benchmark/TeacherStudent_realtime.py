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
from sklearn.model_selection import train_test_split
from threading import Thread
from numba import cuda
import nvidia_smi
import numpy as np
from keras.utils.layer_utils import count_params
from datetime import datetime

"""
Say to GPU to use only the needed amount of memory
"""
GPUs = GPUtil.getGPUs()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

batch_size = 1
net_size = 2000
d = 20000

n_epochs = 1
n_of_class = 10
N = 150
teacher_size = 8
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
if len(sys.argv) != 3:
    print("Teacher-Student benchmark need 2 arguments:")
    print("- Device")
    print("- Changing paramter")
    sys.exit(1)


dev = sys.argv[1]
param = sys.argv[2]

"""
SET DEVICE
"""
if dev == 'cpu':
    de = '/cpu:0'
elif dev == 'gpu':
    if tf.config.list_physical_devices('GPU') == 0:
        print("GPU unavailable :(")
        sys.exit(0)
    de = '/device:GPU:0'

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

# class calculator(Thread):

#     def __init__(self, delay):
#         super(Monitor, self).__init__()
#         self.stopped = False
#         self.delay = delay
#         self.start()

#     def run(self):
#         # self.time = 0
#         # self.count = 0
#         while not self.stopped:
#             self.lat.append(self.time/self.count)
#             self.thr.append(self.count/self.time)


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
        self.gpu_mem_2 = []
        self.gpu_mem_gpuutil = []
        self.gpu_tfmem_curr = []
        self.gpu_tfmem_peak = []
        while not self.stopped:
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_load.append(res.gpu)
            self.gpu_mem.append(res.memory)
            self.gpu_mem_2.append(100*(info.total-info.free)/info.total)
            self.gpu_mem_gpuutil.append(GPUs[0].memoryUtil*100)
            tf_mem_info = tf.config.experimental.get_memory_info('GPU:0')
            self.gpu_tfmem_curr.append(tf_mem_info['current'])
            self.gpu_tfmem_peak.append(tf_mem_info['peak'])
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

def data_loading(d, N, n_of_class): 

    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    '''
    Teacher definition
    '''
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    teacher = Sequential()
    teacher.add(Dense(teacher_size, activation='sigmoid',
                    kernel_initializer=initializer,
                    input_shape=(d,)))
    teacher.add(Dense(n_of_class,
                      activation='softmax'))
    '''
    Random Data generation
    '''
    data = tf.random.normal((N, d), 0, 1)
    labels = teacher.predict(data).argmax(1)
    labels = tf.convert_to_tensor(labels)

    labels = to_categorical(labels, num_classes=n_of_class)
    x_train, x_test, y_train, y_test = data[:int(N*9/10)], data[int(N*9/10):], labels[:int(N*9/10)], labels[int(N*9/10):]
    return (x_train, y_train), (x_test, y_test)

def main(d, batch_size, net_size, n_of_class):
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    stats_file = open("stats_file"+dt_string+".txt", 'w')

    (X_train, Y_train), (X_test, Y_test) = data_loading(d, N, n_of_class)

    """
    Training
    """
    model = Sequential()

    if net_size==0:
        model.add(Dense(n_of_class,
                activation='softmax',
                input_shape=(d,)))
    else:
        model.add(Dense(net_size, activation='sigmoid', input_shape=(d,)))
        model.add(Dense(n_of_class, activation='softmax'))

    loss = keras.losses.CategoricalCrossentropy()
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.05)
    metrics=["accuracy"]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    # model.summary()


    # time_callback = TimeHistory()
    # GPUstats = GPUstatistics_time(0.01)
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size,
            # callbacks=[time_callback],
            verbose=0)
    # GPUstats.stop()
    # training_time = sum(time_callback.batch_times) # total time
    # time_per_sample = training_time/((len(X_train)//batch_size)* batch_size) #time per sample
    # training_sample_per_second = 1./time_per_sample # sample per seconds

    print('TRAINING DONE!')

    """
    Testing Out-of-Sample
    """
    # value = 0
    sample_count = 0
    total_time = 0
    value = True
    # start_total_time = time.time()
    while(value):
        L = []
        # time_callback = TimeHistory()
        for X in X_test: # online prediction (one sample at time)
            X=np.expand_dims(X,0)
            start_time = time.time()
            pred = model(X)
            # pred = model.predict(X,
            #                 batch_size=batch_size
            #                 # callbacks=[time_callback]
            #                 ).argmax(1)
            # pred = model.predict_on_batch(X
            #                 # batch_size=batch_size
            #                 # callbacks=[time_callback]
            #                 ).argmax(1)
            end_time = time.time() - start_time
            total_time += end_time
            sample_count += 1
            latency = total_time/sample_count
            throughput = sample_count/total_time
            L = [str(end_time),',', str(sample_count),',', str(latency),',', str(throughput),',', str(total_time)]
            stats_file.writelines(L)
            stats_file.writelines('\n')


    stats_file.close()




if __name__ == "__main__":
    with tf.device(de):

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        main(d, batch_size, net_size, n_of_class)

