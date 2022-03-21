#!/usr/bin/env python

# Copyright 2021-2022 Open ForBC for the benefit of INFN.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
# - Alessio Borriero <aleborri97@gmail.com>, 2021-2022
# - Daniele Monteleone <daniele.monteleone@to.infn.it>, 2022
# - Gabriele Gaetano Fronze' <gabriele.fronze@to.infn.it>, 2022

import time
import sys
from datetime import datetime
import argparse
import signal
import nvidia_smi
import GPUtil
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, Dense, Input, BatchNormalization


"""
Global variables definition
"""
batch_size = 1
n_epochs = 10
n_epochs_training = 10
n_of_class = 10
N = 150
teacher_size = 8
gpu_performance_sampling_time = 1
n_epochs_feature_extraction = 1
n_epochs_fine_tuning = 1


class GPUstatistics(keras.callbacks.Callback):
    """
    A set of custom Keras callbacks to monitor Nvidia GPUs load
    """

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


class TimeHistory(keras.callbacks.Callback):
    """
    A set of custom Keras callbacks to monitor Nvidia GPUs compute time
    """

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

def resize_images(image, label):
    """
    Data preprcessing method
    """
    image = tf.image.resize(image, size=(224,224)) # resize
    image = tf.cast(image, dtype=tf.float32)    # change data type
    return image, label

def load_FOOD101_data():
    """
    Loading and shaping the standard CIFAR dataset
    """
    (train_data, test_data), ds_info = tfds.load("food101",
                                                 data_dir="FoodVision_dataset",
                                                 split=["train", "validation"],
                                                 shuffle_files=False,
                                                 with_info=True,
                                                 as_supervised=True)

    train_data = train_data.map(map_func=resize_images, num_parallel_calls=tf.data.AUTOTUNE)
    # batch size is small cause images are big and I want to avoid a GPU overload
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    #test_data doesn't need to be shuffled
    test_data = test_data.map(map_func=resize_images, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, test_data


# def create_CIFAR_model():
#     model = keras.models.Sequential()
#     model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
#                                   input_shape=(32, 32, 3)))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dense(10))
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])

#     return model


def feature_extraction_benchmark(batch_size):
    """
    Feature extraction benchmark evaluating number of training inputs processed per second
    """
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    stats_file = open("stats_file_training"+dt_string+".txt", 'w')

    train_data, test_data = load_FOOD101_data()

    """
    Load EfficientNetB0 pre-trained model and add our own output layer
    """
    # To use our own output
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
    # To do only feature extraction and not fine-tuning
    base_model.trainable = False
    
    inputs = Input(shape = (224,224,3), name='inputLayer')
    x = base_model(inputs, training = False) # We not fine tune the model yet
    x = GlobalAveragePooling2D(name='poolingLayer')(x)
    x = Dense(101, name='outputLayer')(x)
    outputs = Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)

    model = tf.keras.Model(inputs, outputs, name = "FeatureExtractionModel")
    model.summary()

    """
    Training - Feature Extraction
    """
    time_callback = TimeHistory()
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam())
    hist_model = model.fit(train_data,
                           epochs = n_epochs_feature_extraction,
                           steps_per_epoch=len(train_data),
                           callbacks=[time_callback], verbose=1)

    training_time = sum(time_callback.batch_times)  # total time
    time_per_epoch = training_time / n_epochs_training
    time_per_batch = time_per_epoch / (len(train_images)//batch_size)  # time per batch
    time_per_sample = time_per_epoch / len(train_images)  # time per sample
    training_sample_per_second = 1./time_per_sample  # sample per seconds

    L = [str(training_time), ',', str(time_per_batch), ',', str(time_per_sample),
         ',', str(training_sample_per_second)]
    stats_file.writelines(L)

    print('FEATURE EXTRACTION COMPLETED!')
    stats_file.close()

def fine_tuning_benchmark(batch_size):
    """
    Fine tuning benchmark evaluating number of training inputs processed per second
    """
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    stats_file = open("stats_file_training"+dt_string+".txt", 'w')

    train_data, test_data = load_FOOD101_data()

    """
    Load EfficientNetB0 pre-trained model and add our own output layer
    """
    # To use our own output
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
    # To do only feature extraction and not fine-tuning
    base_model.trainable = False
    
    inputs = Input(shape = (224,224,3), name='inputLayer')
    x = base_model(inputs, training = False) # We not fine tune the model yet
    x = GlobalAveragePooling2D(name='poolingLayer')(x)
    x = Dense(101, name='outputLayer')(x)
    outputs = Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)

    model = tf.keras.Model(inputs, outputs, name = "FeatureExtractionModel")
    model.summary()

    """
    Feature extraction without measuring its performances to prepare for inference
    """
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam())
    hist_model = model.fit(train_data,
                           epochs = n_epochs_feature_extraction,
                           steps_per_epoch=len(train_data))


    """
    Training - Fine Tuning
    """
    # Let's allow fine tuning
    base_model.trainable = True
    # I don't want batch normalization layer to be trainable
    for layer in model.layers[1].layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    model.summary()
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              # I have to reduce learning rate during fine tuning
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
   
    time_callback = TimeHistory()
    hist_model = model.fit(train_data,
                 epochs = n_epochs_fine_tuning,
                 steps_per_epoch=len(train_data),
                 # Start from the final epoch of the feature extraction step
                 initial_epoch=hist_model.epoch[-1],
                 callbacks=[time_callback])
    
    training_time = sum(time_callback.batch_times)  # total time
    time_per_epoch = training_time / n_epochs_training
    time_per_batch = time_per_epoch / (len(train_images)//batch_size)  # time per batch
    time_per_sample = time_per_epoch / len(train_images)  # time per sample
    training_sample_per_second = 1./time_per_sample  # sample per seconds

    L = [str(training_time), ',', str(time_per_batch), ',', str(time_per_sample),
         ',', str(training_sample_per_second)]
    stats_file.writelines(L)

    print('FINE TUNING COMPLETED!')
    stats_file.close()


def inference_benchmark():
    """
    Inference benchmark evaluating number of Out-of-Sample inputs processed per second
    """


    train_data, test_data = load_FOOD101_data()

    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    stats_file = open("stats_file_inference"+dt_string+".txt", 'w')

    """
    Load EfficientNetB0 pre-trained model and add our own output layer
    """
    # To use our own output
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
    # To do only feature extraction and not fine-tuning
    base_model.trainable = False
    
    inputs = Input(shape = (224,224,3), name='inputLayer')
    x = base_model(inputs, training = False) # We not fine tune the model yet
    x = GlobalAveragePooling2D(name='poolingLayer')(x)
    x = Dense(101, name='outputLayer')(x)
    outputs = Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)

    model = tf.keras.Model(inputs, outputs, name = "FeatureExtractionModel")
    model.summary()

    """
    Feature extraction without measuring its performances to prepare for inference
    """
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam())
    hist_model = model.fit(train_data,
                           epochs = n_epochs_feature_extraction,
                           steps_per_epoch=len(train_data))


    """
    Fine Tuning without measuring its performances to prepare for inference
    """
    # Let's allow fine tuning
    base_model.trainable = True
    # I don't want batch normalization layer to be trainable
    for layer in model.layers[1].layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    model.summary()
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              # I have to reduce learning rate during fine tuning
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
   
    hist_model = model.fit(train_data,
                 epochs = n_epochs_fine_tuning,
                 steps_per_epoch=len(train_data),
                 # Start from the final epoch of the feature extraction step
                 initial_epoch=hist_model.epoch[-1])

    """
    Performing inference on Out-of-Sample multiple times to obtain average performance
    """
    sample_count = 0
    total_time = 0
    keep_running = True

    def handler(foo, bar):
        """
        An handler to catch Ctrl-C for graceful exit
        """
        global keep_running
        keep_running = False

    signal.signal(signal.SIGINT, handler)

    while(keep_running):
        L = []

        for batch in train_data: # online prediction (one sample at time)
            for IMG in batch[0]:
                IMG = np.expand_dims(IMG,0)
                start_time = time.time()
                _ = model(IMG)
                end_time = time.time() - start_time
                total_time += end_time
                sample_count += 1
                latency = total_time/sample_count
                throughput = sample_count/total_time
                L = [str(end_time), ',', str(sample_count), ',', str(
                    latency), ',', str(throughput), ',', str(total_time)]
                stats_file.writelines(L)
                stats_file.writelines('\n')

    print('INFERENCE COMPLETED!')
    stats_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A ML CIFAR benchmark")
    parser.add_argument("device_type",
                        choices=["gpu", "cpu"],
                        default="gpu")
    parser.add_argument("mode",
                        choices=["fine_tuning","feature_extraction", "inference"],
                        default="inference")
    parser.add_argument("gpu_index", default=0, nargs='?')

    args = parser.parse_args()
    dev = args.device_type
    mode = args.mode
    gpu_index = args.gpu_index

    """
    SET DEVICE
    """
    if dev == 'cpu':
        de = '/cpu:0'
    elif dev == 'gpu':
        """
        Checking GPU availability
        """
        if tf.config.list_physical_devices('GPU') == 0:
            print("GPU unavailable. Aborting.")
            sys.exit(0)

        """
        Telling to the used GPU to automatically increase the amount of used memory
        """
        GPUs = GPUtil.getGPUs()
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            if len(gpus) >= gpu_index+1:
                tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            else:
                print(f"GPU {gpu_index} not found. Aborting.")
                sys.exit(0)
        else:
            print("No GPU found. Aborting.")
            sys.exit(0)

        de = f'/device:GPU:{gpu_index}'

    with tf.device(de):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)

        if mode == 'fine_tuning':
            feature_extraction_benchmark(batch_size)
        elif mode == 'feature_extraction':
            fine_tuning_benchmark(batch_size)
        elif mode == 'inference':
            inference_benchmark()
