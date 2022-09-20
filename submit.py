import pandas as pd
import numpy as np
import os
import shutil
import random as python_random
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import time
import joblib
import tensorflow_probability as tfp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import argparse
import sys
from tqdm import tqdm
# sys.path.insert(0, 'data/')
# sys.path.insert(0, './model/')
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Activation

tf.random.set_seed(2022)
np.random.seed(2022)
python_random.seed(2022)

from data.load_data_train import * 
from data.process_missing_data import *
from model.loss_mertric import *
from model.model_encode_decode import *
from model.visualize import *

import yaml

parser = argparse.ArgumentParser(description='Air Quality Forecasting Challenge')
parser.add_argument('--path_save_submit', default='./submit/', type=str, help='path save file result csv')
parser.add_argument('--path_data_test', default='./public-test/input/', type=str, help='path folder publich test')
parser.add_argument('--conf_model', default='./save_model/130822_12/model_save.yml', type=str, help='name folder model to save')
args = parser.parse_args()

if __name__ == '__main__':
 
    with open(args.conf_model) as f:
        CONFIG_MODEL = yaml.safe_load(f)
    
    name_model = CONFIG_MODEL['name']
    FOLDER_SUBMIT = args.path_save_submit + name_model
    PUBLIC_TEST_INPUT_PATH = args.path_data_test

    COL_NAME = CONFIG_MODEL['col']
    X_min = CONFIG_MODEL['min_x']
    X_min= np.array(X_min)
    X_max = CONFIG_MODEL['max_x']
    X_max= np.array(X_max)
    Y_min = CONFIG_MODEL['min_y']
    Y_min= np.array(Y_min)
    Y_max = CONFIG_MODEL['max_y']
    Y_max= np.array(Y_max)
    PATH_MODEL = CONFIG_MODEL['path_model']
    list_imputer = CONFIG_MODEL['path_list_imputer']
    base_model = CONFIG_MODEL['base_model']
    step_missing = CONFIG_MODEL['Args']['step_missing']
    past_ = CONFIG_MODEL['Args']['past']
    step = CONFIG_MODEL['Args']['step']
    step_2 = int(past_/ step)

    model = keras.models.load_model(PATH_MODEL, custom_objects={"mdape": mdape,'loss_custom':loss_custom })
    dict_data_submit = load_data_test(PUBLIC_TEST_INPUT_PATH, COL_NAME)

    for k,v in tqdm(dict_data_submit.items()):
        v1 = fill_missing_knn(v,step_missing, list_imputer)
        v1 = (v1 - X_min)/ (X_max - X_min)
        if int(step) > 1:
            b = np.zeros(shape = (24, step_2, len(COL_NAME)))
            seq1 = [[i+ j for j in [0,24,48,72,96,120,144]] for i in range(24)]
            for i in seq1:
                b[i[0],:,:] = v1[i]
            predict = model.predict(b)
            predict = np.squeeze(predict, axis= 1)
        else:
            v1 = np.expand_dims(v1, axis=0)
            predict = model.predict(v1)
            predict = np.squeeze(predict, axis= 0)

        predict = (predict * (Y_max - Y_min)) + Y_min

        if not os.path.exists(FOLDER_SUBMIT):
            os.mkdir(FOLDER_SUBMIT)
        if not os.path.exists(FOLDER_SUBMIT + k):
            os.mkdir(FOLDER_SUBMIT + k)
        for i in range(1,5):
            name_submit = FOLDER_SUBMIT + str(k) + '/res_'+ k+ '_' +str(i) + '.csv'
            pd.DataFrame(predict[:,i-1], columns= ['PM2.5']).to_csv(name_submit, index= False)
    
    shutil.make_archive('submit' + name_model[:-1] , 'zip', FOLDER_SUBMIT)