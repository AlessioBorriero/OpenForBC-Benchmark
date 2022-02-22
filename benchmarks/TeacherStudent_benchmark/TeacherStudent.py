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

"""
Say to GPU to use only the needed amount of memory
"""
GPUs = GPUtil.getGPUs()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

dims = [10,20,30,40,50]
batch_sizes = [50,200,400,600,800,1000,1200,1400,1800,2000]
# net_sizes = [10,100,300,500,700,900,1000,
#               2000,4000,6000,8000,10000,30000,50000,
#               60000,80000,100000]
net_sizes = [10,500,700,900,1000,
              2000,4000,6000,8000,10000,30000,50000,
              60000,80000,100000]


d = 1000
# batch_size = 32
batch_size = 1
net_size = 20
n_epochs = 100
n_of_class = 10
N = 2000
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
    x_train, x_test, y_train, y_test = data[:9000], data[9000:], labels[:9000], labels[9000:]
    return (x_train, y_train), (x_test, y_test)

def main(d, batch_size, net_size, n_of_class):

    (X_train, Y_train), (X_test, Y_test) = data_loading(d, N, n_of_class)

    """
    Training
    """
    for i in range(50):
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


        time_callback = TimeHistory()
        GPUstats = GPUstatistics_time(0.01)
        model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size,
                callbacks=[time_callback],
                verbose=0)
        GPUstats.stop()
        training_time = sum(time_callback.batch_times) # total time
        time_per_sample = training_time/((len(X_train)//batch_size)* batch_size) #time per sample
        training_sample_per_second = 1./time_per_sample # sample per seconds

        gpu_loads_train[len(gpu_loads_train)-1].append(GPUstats.gpu_load)
        # gpu_mems_train.append(np.asarray(GPUstats.gpu_mem))
        # gpu_mems_2_train.append(np.asarray(GPUstats.gpu_mem_2))
        # gpu_mems_gpuutil_train.append(np.asarray(GPUstats.gpu_mem_gpuutil))
        # gpu_tfmem_peak_train.append(np.asarray(GPUstats.gpu_tfmem_peak))
        gpu_tfmem_curr_train[len(gpu_tfmem_curr_train)-1].append(np.asarray(GPUstats.gpu_tfmem_curr))
        times_train[len(times_train)-1].append(training_time)
        sample_per_second_train[len(sample_per_second_train)-1].append(training_sample_per_second)


    """
    Testing In-Sample
    """
    for i in range(50):
        load = []
        mem = []
        time = []
        sample_per_second = []
        for _ in range(50):
            time_callback = TimeHistory()

            GPUstats = GPUstatistics_time(0.01)
            pred = model.predict(X_train, 
                            batch_size=batch_size,
                            callbacks=[time_callback]).argmax(1)
            GPUstats.stop()

            testing_time_insample = sum(time_callback.batch_times)
            gpu_loads_testin = GPUstats.gpu_load
            gpu_mems_testin = GPUstats.gpu_mem
            time_per_sample = testing_time_insample/len(X_train)
            test_insample_sample_per_second = 1./time_per_sample

            load.append(GPUstats.gpu_load)
            mem.append(GPUstats.gpu_tfmem_curr)
            time.append(testing_time_insample)
            sample_per_second.append(test_insample_sample_per_second)

        gpu_loads_infin[len(gpu_loads_infin)-1].append(load)
        # gpu_mems_infin[len(gpu_mems_infin)-1].append(np.asarray(GPUstats.gpu_mem))
        # gpu_mems_2_infin[len(gpu_mems_2_infin)-1].append(np.asarray(GPUstats.gpu_mem_2))
        # gpu_mems_gpuutil_infin[len(gpu_mems_gpuutil_infin)-1].append(np.asarray(GPUstats.gpu_mem_gpuutil))
        # gpu_tfmem_peak_infin[len(gpu_tfmem_peak_infin)-1].append(np.asarray(GPUstats.gpu_tfmem_peak))
        gpu_tfmem_curr_infin[len(gpu_tfmem_curr_infin)-1].append(mem)
        times_infin[len(times_infin)-1].append(time)
        sample_per_second_infin[len(sample_per_second_infin)-1].append(sample_per_second)

    """
    Testing Out-of-Sample
    """
    for i in range(50):
        load = []
        mem = []
        time = []
        sample_per_second = []
        for _ in range(50):
            time_callback = TimeHistory()
            GPUstats = GPUstatistics_time(0.01)

            pred = model.predict(X_test,
                            batch_size=batch_size,
                            callbacks=[time_callback]).argmax(1)
            GPUstats.stop()
            
            testing_time_outofsample = sum(time_callback.batch_times)
            gpu_loads_testout = GPUstats.gpu_load
            gpu_mems_testout = GPUstats.gpu_mem
            accuracy = accuracy_score(Y_test.argmax(1),
                                    pred, normalize=True)
            time_per_sample = testing_time_outofsample/len(X_test)
            test_outofsample_sample_per_second = 1./time_per_sample

            load.append(GPUstats.gpu_load)
            mem.append(GPUstats.gpu_tfmem_curr)
            time.append(testing_time_insample)
            sample_per_second.append(test_insample_sample_per_second)

        gpu_loads_infout[len(gpu_loads_infout)-1].append(load)
        # gpu_mems_infin[len(gpu_mems_infin)-1].append(np.asarray(GPUstats.gpu_mem))
        # gpu_mems_2_infin[len(gpu_mems_2_infin)-1].append(np.asarray(GPUstats.gpu_mem_2))
        # gpu_mems_gpuutil_infin[len(gpu_mems_gpuutil_infin)-1].append(np.asarray(GPUstats.gpu_mem_gpuutil))
        # gpu_tfmem_peak_infin[len(gpu_tfmem_peak_infin)-1].append(np.asarray(GPUstats.gpu_tfmem_peak))
        gpu_tfmem_curr_infout[len(gpu_tfmem_curr_infout)-1].append(mem)
        times_infout[len(times_infout)-1].append(time)
        sample_per_second_infout[len(sample_per_second_infout)-1].append(sample_per_second)


if __name__ == "__main__":
    with tf.device(de):
        if   param=='data_dims':
            gpu_loads_train = []
            gpu_tfmem_curr_train = []
            times_train = []
            sample_per_second_train = []

            gpu_loads_infin = []
            gpu_tfmem_curr_infin = []
            times_infin = []
            sample_per_second_infin = []

            gpu_loads_infout = []
            gpu_tfmem_curr_infout = []
            times_infout = []
            sample_per_second_infout = []

            for d in dims:
                gpu_loads_train.append([])
                gpu_tfmem_curr_train.append([])
                times_train.append([])
                sample_per_second_train.append([])

                gpu_loads_infin.append([])
                gpu_tfmem_curr_infin.append([])
                times_infin.append([])
                sample_per_second_infin.append([])

                gpu_loads_infout.append([])
                gpu_tfmem_curr_infout.append([])
                times_infout.append([])
                sample_per_second_infout.append([])

                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                main(d, batch_size, net_size, n_of_class)

            np.savez('gpu_load_train'+'_batch',gpu_loads_train)
            np.savez('gpu_mem_train'+'_batch',gpu_tfmem_curr_train)
            np.savez('time_train'+'_batch',times_train)
            np.savez('sample_per_second_train'+'_batch',sample_per_second_train)

            np.savez('gpu_load_infin'+'_batch',gpu_loads_infin)
            np.savez('gpu_mem_infin'+'_batch',gpu_tfmem_curr_infin)
            np.savez('time_infin'+'_batch',times_infin)
            np.savez('sample_per_second_infin'+'_batch',sample_per_second_infin)

            np.savez('gpu_load_infout'+'_batch',gpu_loads_infout)
            np.savez('gpu_mem_infout'+'_batch',gpu_tfmem_curr_infout)
            np.savez('time_infout'+'_batch',times_infout)
            np.savez('sample_per_second_infout'+'_batch',sample_per_second_infout)

        elif param=='batch_size':
            gpu_loads_train = []
            gpu_tfmem_curr_train = []
            times_train = []
            sample_per_second_train = []

            gpu_loads_infin = []
            gpu_tfmem_curr_infin = []
            times_infin = []
            sample_per_second_infin = []

            gpu_loads_infout = []
            gpu_tfmem_curr_infout = []
            times_infout = []
            sample_per_second_infout = []

            for batch_size in batch_sizes:
                gpu_loads_train.append([])
                gpu_tfmem_curr_train.append([])
                times_train.append([])
                sample_per_second_train.append([])

                gpu_loads_infin.append([])
                gpu_tfmem_curr_infin.append([])
                times_infin.append([])
                sample_per_second_infin.append([])

                gpu_loads_infout.append([])
                gpu_tfmem_curr_infout.append([])
                times_infout.append([])
                sample_per_second_infout.append([])

                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                main(d, batch_size, net_size, n_of_class)

            np.savez('gpu_load_train'+'_batch',gpu_loads_train)
            np.savez('gpu_mem_train'+'_batch',gpu_tfmem_curr_train)
            np.savez('time_train'+'_batch',times_train)
            np.savez('sample_per_second_train'+'_batch',sample_per_second_train)

            np.savez('gpu_load_infin'+'_batch',gpu_loads_infin)
            np.savez('gpu_mem_infin'+'_batch',gpu_tfmem_curr_infin)
            np.savez('time_infin'+'_batch',times_infin)
            np.savez('sample_per_second_infin'+'_batch',sample_per_second_infin)

            np.savez('gpu_load_infout'+'_batch',gpu_loads_infout)
            np.savez('gpu_mem_infout'+'_batch',gpu_tfmem_curr_infout)
            np.savez('time_infout'+'_batch',times_infout)
            np.savez('sample_per_second_infout'+'_batch',sample_per_second_infout)

        elif param=='net_size':
            gpu_loads_train = []
            gpu_tfmem_curr_train = []
            times_train = []
            sample_per_second_train = []

            gpu_loads_infin = []
            gpu_tfmem_curr_infin = []
            times_infin = []
            sample_per_second_infin = []

            gpu_loads_infout = []
            gpu_tfmem_curr_infout = []
            times_infout = []
            sample_per_second_infout = []

            for net_size in net_sizes:
                gpu_loads_train.append([])
                gpu_tfmem_curr_train.append([])
                times_train.append([])
                sample_per_second_train.append([])

                gpu_loads_infin.append([])
                gpu_tfmem_curr_infin.append([])
                times_infin.append([])
                sample_per_second_infin.append([])

                gpu_loads_infout.append([])
                gpu_tfmem_curr_infout.append([])
                times_infout.append([])
                sample_per_second_infout.append([])

                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                main(d, batch_size, net_size, n_of_class)
                
            np.savez('gpu_load_train'+'_netsize',gpu_loads_train)
            np.savez('gpu_mem_train'+'_netsize',gpu_tfmem_curr_train)
            np.savez('time_train'+'_netsize',times_train)
            np.savez('sample_per_second'+'_netsize',sample_per_second_train)

            np.savez('gpu_load_infin'+'_netsize',gpu_loads_infin)
            np.savez('gpu_mem_infin'+'_netsize',gpu_tfmem_curr_infin)
            np.savez('time_infin'+'_netsize',times_infin)
            np.savez('sample_per_second_infin'+'_netsize',sample_per_second_infin)

            np.savez('gpu_load_infout'+'_netsize',gpu_loads_infout)
            np.savez('gpu_mem_infout'+'_netsize',gpu_tfmem_curr_infout)
            np.savez('time_infout'+'_netsize',times_infout)
            np.savez('sample_per_second_infout'+'_netsize',sample_per_second_infout)

