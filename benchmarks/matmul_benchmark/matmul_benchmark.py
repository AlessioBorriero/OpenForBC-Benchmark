#!/usr/bin/env python

import tensorflow as tf
import numpy as np
# import cupy as cp
import sys
import time
import GPUtil
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
from tensorflow.compat.v1 import global_variables_initializer, Session, enable_eager_execution, disable_eager_execution

shapes = [100,500,1000,2000,5000,10000,20000]
reps = [5000,3000,2000,1000,400,300,200]

# READ ARGS FROM COMMAND LINE
# Raise error if correct arguments aren't given
# if len(sys.argv) != 4:
#     print("Matmul benchmark need 3 arguments:")
#     print("- device")
#     print("- matrices first dimension")
#     print("- matrices second dimension")
#     sys.exit(1)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

dev = 'gpu'
GPUs = GPUtil.getGPUs()
# dev = sys.argv[1]
# shape_1 = int(sys.argv[2])
# shape_2 = int(sys.argv[3])

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

# SET DEVICE

if dev == 'cpu':
    d = '/cpu:0'
elif dev == 'gpu':
    if tf.config.list_physical_devices('GPU') == 0:
        print("GPU unavailable :(")
        sys.exit(0)
    d = '/device:GPU:0'

def main(shape, rep):
    sampling_time = 0.01

    shape_1 = shape
    shape_2 = shape

    disable_eager_execution()
    init = global_variables_initializer()
    sess = Session()
    sess.run(init)

    # TENSORFLOW
    tf.random.set_seed(5)
    matrix_1_tf = tf.ones([shape_1, shape_2])
    matrix_2_tf = tf.ones([shape_2, shape_1])
    mult = tf.matmul(matrix_1_tf, matrix_2_tf)

    # to allocate memory inside tf Session
    sess.run(mult)


    # the first operation inside the loop above is slower
    # the gpu take more time to make the operations
    # I don't know why
    for i in range(200):    
        sess.run(init)
        start_time = time.time()
        GPUstats = GPUstatistics_time(sampling_time)
        for _ in range(reps[rep]):
            sess.run(mult)
        GPUstats.stop()
        matmul_time = time.time() - start_time

        gpu_loads[len(gpu_loads)-1].append(GPUstats.gpu_load)
        gpu_mems[len(gpu_mems)-1].append(np.asarray(GPUstats.gpu_mem))
        gpu_mems_2[len(gpu_mems_2)-1].append(np.asarray(GPUstats.gpu_mem_2))
        gpu_mems_gpuutil[len(gpu_mems_gpuutil)-1].append(np.asarray(GPUstats.gpu_mem_gpuutil))
        gpu_tfmem_curr[len(gpu_tfmem_curr)-1].append(np.asarray(GPUstats.gpu_tfmem_curr))
        gpu_tfmem_peak[len(gpu_tfmem_peak)-1].append(np.asarray(GPUstats.gpu_tfmem_peak))
        matmul_times[len(matmul_times)-1].append(matmul_time)
        # print('LOAD {} - len:{} max:{} min:{} mean:{} time:{}'
        #     .format(i,len(GPUstats.gpu_load),np.asarray(GPUstats.gpu_load).max(),
        #             np.asarray(GPUstats.gpu_load).min(),np.asarray(GPUstats.gpu_load).mean(),matmul_time))
        # print('MEM {} - len:{} max:{} min:{} mean:{} time:{}'
        #     .format(i,len(GPUstats.gpu_mem),np.asarray(GPUstats.gpu_mem).max(),
        #             np.asarray(GPUstats.gpu_mem).min(),np.asarray(GPUstats.gpu_mem).mean(),matmul_time))


    return 0

if __name__ == "__main__":
    with tf.device(dev):
        gpu_loads = []
        gpu_mems = []
        gpu_mems_2 = []
        gpu_mems_gpuutil = []
        gpu_tfmem_curr = []
        gpu_tfmem_peak = []
        matmul_times = []
        rep = 0
        for shape in shapes:
            gpu_loads.append([])
            gpu_mems.append([])
            gpu_mems_2.append([])
            gpu_mems_gpuutil.append([])
            gpu_tfmem_curr.append([])
            gpu_tfmem_peak.append([])
            matmul_times.append([])
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            main(shape, rep)
            rep += 1
        # print(np.asarray(shape_s).shape,
        #     np.asarray(gpu_loads_tf).shape,
        #     np.asarray(gpu_mems_tf).shape,
        #     np.asarray(time_tf).shape)
 
        np.savez('matmul_gpu_loads', gpu_loads)
        np.savez('matmul_gpu_mems', gpu_mems)
        np.savez('matmul_gpu_mems_gpuutil', gpu_mems_gpuutil)
        np.savez('matmul_gpu_mems_2', gpu_mems_2)
        np.savez('matmul_gpu_tfmem_curr', gpu_tfmem_curr)
        np.savez('matmul_gpu_tfmem_peak', gpu_tfmem_peak)
        np.savez('matmul_times', matmul_times)



