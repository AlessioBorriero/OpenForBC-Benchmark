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

shapes = [10,500,1000,5000,10000,15000,20000,25000,30000,35000,40000,45000]

# READ ARGS FROM COMMAND LINE
# Raise error if correct arguments aren't given
if len(sys.argv) != 4:
    print("Matmul benchmark need 3 arguments:")
    print("- device")
    print("- matrices first dimension")
    print("- matrices second dimension")
    sys.exit(1)

dev = sys.argv[1]
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
        while not self.stopped:
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_load.append(res.gpu)
            self.gpu_mem.append(res.memory)
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

def main(shape):

        shape_1 = shape
        shape_2 = shape

        # TENSORFLOW
        tf.random.set_seed(5)
        matrix_1_tf = tf.ones([shape_1, shape_2])
        matrix_2_tf = tf.ones([shape_2, shape_1])

        print(matrix_1_tf.device, matrix_1_tf.shape)

        # # NUMPY
        # matrix_1_np = np.random.randn(shape_1, shape_2)
        # matrix_2_np = np.random.randn(shape_2, shape_1)

        # CUPY
        # matrix_1_cp = cp.random.normal(size=shape_1)
        # matrix_2_cp = cp.random.normal(size=shape_2)
        
        gpu_load_max = []
        gpu_load_max = []
        for _ in range(10):
            GPUstats = GPUstatistics_time(0.0000001)
            start_time_tf = time.time()
            for __ in range(10):
                tf.linalg.matmul(matrix_1_tf, matrix_2_tf)
            matmul_time_tf = time.time() - start_time_tf
            GPUstats.stop()
            gpu_load_max.append(np.asarray(GPUstats.gpu_load).max())
            gpu_mem_max.append(np.asarray(GPUstats.gpu_mem).max()) 


        # start_time_np = time.time()
        # np.matmul(matrix_1_np, matrix_2_np)
        # multiplication_time_np = time.time() - start_time_np

        # start_time_cp = time.time()
        # cp.matmul(matrix_1_cp, matrix_2_cp)
        # multiplication_time_cp = time.time() - start_time_cp

        shape_s.append(shape)
        gpu_loads_tf.append(np.asarray(gpu_load_max).mean())
        gpu_mems_tf.append(np.asarray(gpu_mem_max).mean())
        time_tf.append(matmul_time_tf)  

    # print("Matrix multiplcation time with numpy: %s s" % multiplication_time_np)
    # print("Matrix multiplcation time with cupy: {} s" .format(multiplication_time_cp))
        print("Matrix multiplcation time shape {}: {} s" .format(shape, matmul_time_tf))
        print("GPU load {}: {} " .format(shape, np.asarray(gpu_load_tf).mean()))
        print("GPU mem {}: {} " .format(shape, np.asarray(gpu_mem_tf).mean()))


        """
        free gpu memory
        """

        # device = cuda.get_current_device()
        # device.reset()

        return 0

if __name__ == "__main__":
    with tf.device(dev):
        shape_s = []
        gpu_loads_tf = []
        gpu_mems_tf = []
        time_tf = []
        for shape in shapes:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            main(shape)
        # print(np.asarray(shape_s).shape,
        #     np.asarray(gpu_loads_tf).shape,
        #     np.asarray(gpu_mems_tf).shape,
        #     np.asarray(time_tf).shape)
        np.savez('matmul_shapes', shape_s)
        np.savez('matmul_gpu_loads_tf', gpu_loads_tf)
        np.savez('matmul_gpu_mems_tf', gpu_mems_tf)
        np.savez('matmul_time_tf', time_tf)



