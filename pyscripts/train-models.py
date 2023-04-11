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

from libs.utils import prepare_fashion, prepare_mnist
from libs.constants import MODELS_FOLDER

# Limit GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
from libs.model_archs import lenetA, fcA

from libs.seeds import load_model_seeds
model_seeds = load_model_seeds()

# define dataset and model architecture
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model_arch')
args= parser.parse_args()

dataset = args.dataset
model_arch = args.model_arch

# set training hyperparameters
batch_size = 32
n_epochs = 10

# global seed
for seed in model_seeds:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # prepare data
    if dataset == "fashion":
        (x_train, y_train), (x_test, y_test) = prepare_fashion()
    elif dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = prepare_mnist()
    else:
        print("Invalid Dataset or Dataset not found")

    # create model
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    n_classes = y_train.shape[1]

    modelarchfuncList = {'lenetA': lenetA, 'fcA': fcA }
    parameters = {'input_shape':input_shape, 'nb_classes':n_classes}

    model = modelarchfuncList[model_arch](**parameters)

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
              validation_split=0.2)

    # evaluate model
    score = model.evaluate(x_test, 
                           y_test, 
                           batch_size=batch_size)
    print("Original Accuracy: ",score)

    # save model
    model_type = dataset + "--" + model_arch
    model_instance = model_type + "-" + str(seed)
    model_filename = model_instance + ".h5"
    model_subdir = pathlib.Path(MODELS_FOLDER / model_arch)
    pathlib.Path(model_subdir).mkdir(parents=True, exist_ok=True)
    model_file = str(pathlib.Path(model_subdir/ model_filename))
    model.save(model_file)
    model.summary()