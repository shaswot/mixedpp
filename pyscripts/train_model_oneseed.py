import os
import sys
import git
import pathlib

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

import libs.model_archs
import libs.utils
from libs.constants import MODELS_FOLDER

# Limit GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# define dataset and model architecture
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train with')
parser.add_argument('--model_arch', type=str, default='fcA', help='NN Model architectures defined in libs/model_archs.py')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--loss_function', type=str, default='sparse_categorical_crossentropy', help='Loss function to use')
args= parser.parse_args()

dataset = args.dataset
model_arch = args.model_arch
seed = args.seed
batch_size = args.batch_size
n_epochs = args.epochs
loss_fn = args.loss_function

# global seed
tf.random.set_seed(seed)
np.random.seed(seed)

# prepare data
dataset_loader = getattr(libs.utils, 'prepare_'+dataset)
(x_train, y_train), (x_test, y_test) = dataset_loader()

# create model
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
n_classes = y_train.shape[1]

model_generator = getattr(libs.model_archs, model_arch)
model = model_generator(input_shape, n_classes)

# compile model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'],
)

# train model
model.fit(x_train, 
          y_train, 
          batch_size=batch_size, 
          epochs=n_epochs,
          validation_split=0.2,
          verbose=False)

# save model
model_type = dataset + "--" + model_arch
model_instance = model_type + "-" + str(seed)
model_filename = model_instance + ".h5"
model_subdir = pathlib.Path(MODELS_FOLDER / model_arch)
pathlib.Path(model_subdir).mkdir(parents=True, exist_ok=True)
model_file = str(pathlib.Path(model_subdir/ model_filename))
model.save(model_file)
# model.summary()
