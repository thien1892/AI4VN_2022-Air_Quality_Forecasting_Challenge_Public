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
from tqdm import tqdm
import sys
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
from model.model_24_hours import *
from model.visualize import *
from data.data_model import *

import yaml

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Air Quality Forecasting Challenge')
parser.add_argument('--conf_data', default='./CONFIG/model_thien.yml', type=str, help='path of config data')
parser.add_argument('--name_model', default='thien/', type=str, help='name folder model to save')
parser.add_argument('--base_model', default='model_encode_decode', type=str, help='base model')
parser.add_argument('--batch_size', default=32, type=int, help='batch size to process data to model')
# parser.add_argument('--batch_size_model', default=32, type=int, help='batch size train model')
parser.add_argument('--drop_out', default=0.15, type=float, help='drop out')
parser.add_argument('--epochs', default=500, type=int, help='epochs to train')
parser.add_argument('--future', default=24, type=int, help='future step time to predict')
parser.add_argument('--load_weight_imputer', default=False, type=str2bool , help='if true load weight was fitted')
parser.add_argument('--imputer_choose', default='XGB', type=str, help='imputer model to process missing data KNN, XGB, Iterative')
parser.add_argument('--keep_col_percent', default=20, type=int, help='percent missing data, colums will keep when missing < number of percent')
parser.add_argument('--learning_rate', default=7e-4, type=float, help='learning rate')
parser.add_argument('--max_iter', default=10, type=int, help='max iter in iterative imputer')
parser.add_argument('--n_Dense', default=128, type=int, help='n Dense in Layer Dense')
parser.add_argument('--n_LSTM', default=128, type=int, help='n LSTM in Layer LSTM')
parser.add_argument('--n_neighbors', default=3, type=int, help='n_neighbors in KNN Imputer')
parser.add_argument('--past', default=168, type=int, help='past step time')
parser.add_argument('--sequence_stride', default=24, type=int, help='sequence_stride process data train X')
parser.add_argument('--sequence_stride_val', default=24, type=int, help='sequence_stride process data train Y')
parser.add_argument('--split_fraction', default=0.8, type=float, help='split data to train, val')
parser.add_argument('--step', default=1, type=int, help='step to process data to train')
parser.add_argument('--step_missing', default=1, type=int, help='step parts process data missing')
parser.add_argument('--stop_early', default=30, type=int, help='stop early to train')
parser.add_argument('--monitor', default='val_loss', type=str, help='monitor model val_loss or val_mdape')
parser.add_argument('--loss', default='mae', type=str, help='monitor model loss mae or custom')

args = parser.parse_args()

if __name__ == '__main__':
    # print(args.conf_path)
    with open(args.conf_data) as f:
        CONFIG_MODEL = yaml.safe_load(f)
    
    PATH_SAVE_MODEL = CONFIG_MODEL['PATH_SAVE_MODEL'] + args.name_model
    FOLDER_SUBMIT = CONFIG_MODEL['FOLDER_SUBMIT'] + args.name_model

    PATH_LOCATION_INPUT = CONFIG_MODEL['PATH_LOCATION_INPUT']
    PATH_LOCATION_OUTPUT = CONFIG_MODEL['PATH_LOCATION_OUTPUT']
    PUBLIC_TEST_INPUT_PATH = CONFIG_MODEL['PUBLIC_TEST_INPUT_PATH']
    TRAIN_INPUT_PATH = CONFIG_MODEL['TRAIN_INPUT_PATH']
    TRAIN_OUTPUT_PATH = CONFIG_MODEL['TRAIN_OUTPUT_PATH']

    if not os.path.exists(PATH_SAVE_MODEL):
        os.makedirs(PATH_SAVE_MODEL)

    if not os.path.exists(FOLDER_SUBMIT):
        os.makedirs(FOLDER_SUBMIT)
    
    # 1. LOAD DATA
    print('1.LOAD DATA:')
    df_total_train_input = load_data_train(PATH_LOCATION_INPUT, TRAIN_INPUT_PATH)
    # df_total_train_input.to_csv(PATH_SAVE_MODEL+'train_input.csv')
    df_total_train_output = load_data_train(PATH_LOCATION_OUTPUT, TRAIN_OUTPUT_PATH)
    # df_total_train_output.to_csv(PATH_SAVE_MODEL+'train_output.csv')
    df_total_train_total = pd.merge(df_total_train_output,df_total_train_input, left_index=True, right_index=True)
    COL_NAME = check_feature(df_total_train_input, args.keep_col_percent)
    df_total_public_test = load_data_public_test(PUBLIC_TEST_INPUT_PATH, COL_NAME)
    # df_total_public_test.to_csv(PATH_SAVE_MODEL+'public_test.csv')
    df_total_input_imputer = pd.concat((df_total_train_input, df_total_public_test))

    # 2. PROCESS MISSING DATA
    print('2.PROCESS MISSING DATA:')
    print('This step may be take a long time, please wait until complete!!!')
    # train_split = int(args.split_fraction * int(df_total_train_input.shape[0]))
    df_total_train_input = df_total_train_input[COL_NAME]
    df_total_input_imputer = df_total_input_imputer[COL_NAME]
    # df_total_train_input_imputer = df_total_train_input[:train_split]
    df_total_train_total = df_total_train_total[df_total_train_output.columns.to_list() + COL_NAME]
    # df_total_train_total_imputer = df_total_train_total[:train_split]

    if not args.load_weight_imputer:
        if args.imputer_choose == 'XGB':
            list_imputer = XGB_Imputer(df_total_input_imputer.values, args.step_missing, args.max_iter, PATH_SAVE_MODEL+'model_imputer_input/')
            list_imputer_out = XGB_Imputer(df_total_train_total.values, args.step_missing, args.max_iter, PATH_SAVE_MODEL+'model_imputer_output/')
        if args.imputer_choose == 'KNN':
            list_imputer = KNN_Imputer(df_total_input_imputer.values, args.step_missing, args.n_neighbors, PATH_SAVE_MODEL+'model_imputer_input/')
            list_imputer_out = KNN_Imputer(df_total_train_total.values, args.step_missing, args.n_neighbors, PATH_SAVE_MODEL+'model_imputer_output/')
        elif args.imputer_choose == 'Iterative':
            list_imputer = Iterative_Imputer(df_total_input_imputer.values, args.step_missing, args.max_iter, PATH_SAVE_MODEL+'model_imputer_input/')
            list_imputer_out = Iterative_Imputer(df_total_train_total.values, args.step_missing, args.max_iter, PATH_SAVE_MODEL+'model_imputer_output/')
    else:
        if args.imputer_choose == 'XGB':
            list_imputer = ['save_model/130822_12/model_imputer_input/XGB_imputer_slice_0.pkl']
            list_imputer_out = ['save_model/130822_12/model_imputer_output/XGB_imputer_slice_0.pkl']

    data_fill_input = fill_missing_knn(df_total_train_input.values,args.step_missing, list_imputer)
    data_fill_public_test = fill_missing_knn(df_total_public_test.values,args.step_missing, list_imputer)
    data_fill_output = fill_missing_knn(df_total_train_total.values,args.step_missing, list_imputer_out)

    dataset_train_input = pd.DataFrame(data_fill_input, columns= COL_NAME)
    dataset_train_input.to_csv(PATH_SAVE_MODEL+'input_fill_missing.csv')
    dataset_public_test  = pd.DataFrame(data_fill_public_test, columns= COL_NAME)
    dataset_public_test.to_csv(PATH_SAVE_MODEL+'public_test_fill_missing.csv')
    dataset_train_output = pd.DataFrame(data_fill_output[:,:12], columns= df_total_train_output.columns)
    dataset_train_output.to_csv(PATH_SAVE_MODEL+'output_fill_missing.csv')

    # 3. Prepare data for MODEL
    print('3. PREPARE DATA FOR MODEL:')
    train_split = int(args.split_fraction * int(df_total_train_input.shape[0]))
    print(f'train split: {train_split}')
    Y = dataset_train_output.iloc[:, [0,3,6,9]]
    Y, Y_min, Y_max = normalize(Y.values, train_split)
    Y = pd.DataFrame(Y)
    X, X_min, X_max = normalize(dataset_train_input.values, train_split)
    X = pd.DataFrame(X)
    train_X = X.loc[0 : train_split - 1]
    val_X = X.loc[train_split:]

    ## 3.1. Datatrain
    start = args.past
    end = args.future + train_split
    sequence_length_x = int(args.past / args.step)
    sequence_length_y = int(args.future/ args.step)
    print(f'train start: {start}, end: {end}')
    train_Y = Y.iloc[start:end]

    train_dataset = create_tf_data(train_X , 
                    train_Y ,
                    sequence_stride = args.sequence_stride,
                    sequence_length_x = sequence_length_x,
                    sequence_length_y = sequence_length_y,
                    sampling_rate= args.step,
                    batch_size = args.batch_size,
                    shuffle= True)
    for batch in train_dataset.take(1):
        train_input, train_output = batch
        print(f'train inputs shape: {train_input.shape}; train outputs shape: {train_output.shape}')
        break

    ## 3.2. Data val
    x_end = len(val_X) - args.future
    label_start = train_split + args.past

    print(f'val label start: {label_start}, x end: {x_end}')
    val_Y = Y.iloc[label_start:]
    val_X_ = val_X.iloc[:x_end]
    print(f'val X shape: {val_X.shape}, val Y shape: {val_Y.shape}')

    val_dataset = create_tf_data(val_X_ , 
                val_Y ,
                sequence_stride = args.sequence_stride,
                sequence_length_x = sequence_length_x,
                sequence_length_y = sequence_length_y,
                sampling_rate= args.step,
                batch_size = args.batch_size,
                shuffle=False)

    for batch in val_dataset.take(1):
        val_input, val_output = batch
        print(f'train inputs shape: {val_input.shape}; train outputs shape: {val_output.shape}')
        break
    
    # 4.MODEL
    n_steps_in, n_features =  val_input.shape[1:]
    n_steps_out, n_features_out =  val_output.shape[1:]

    if args.base_model == 'model_encode_decode':
        model = model_encode_decode(
                n_steps_in = n_steps_in,
                n_features = n_features,
                n_steps_out = n_steps_out,
                n_features_out = n_features_out,
                n_LSTM = args.n_LSTM,
                n_Dense = args.n_Dense,
                drop_out = args.drop_out,
                )
    elif args.base_model == 'model_24_hours':
        model = model_24_hours(
                input_shape_0 =  n_steps_in,
                input_shape_1 = n_features,
                output_shape_0 =  n_steps_out,
                output_shape_1 =  n_features_out,
                drop_out = args.drop_out,
                n_LSTM = args.n_LSTM,
                n_Dense = args.n_Dense,
                )
    # plot_model(model, show_shapes=True, to_file=PATH_SAVE_MODEL+'model5.png')

    print(model.summary())
    metrics = [mdape,
            # tf.keras.metrics.MeanAbsolutePercentageError(name = 'MAPE'),
            tf.keras.metrics.MeanAbsoluteError(name = 'MAE'),
            tf.keras.metrics.RootMeanSquaredError(name = 'RMSE')
            ]
    optimizer = keras.optimizers.Adam(learning_rate= args.learning_rate, clipnorm=1)

    if args.loss == 'mae':
        model.compile(loss= 'mae',
                    optimizer=optimizer,
                    metrics= metrics)
    elif args.loss == 'custom':
        model.compile(loss= loss_custom,
                    optimizer=optimizer,
                    metrics= metrics)     

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        PATH_SAVE_MODEL+"my_checkpoint.h5", save_best_only=True,monitor= args.monitor , verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience= args.stop_early, monitor= args.monitor)
    print('4.TRAIN MODEL:')
    history = model.fit(train_dataset, epochs= args.epochs,
            validation_data= val_dataset,
            callbacks=[early_stopping, model_checkpoint], verbose= 1, batch_size= args.batch_size)
    
    # Evaluate MOdel
    print('5.EVALUATE MODEL:')
    visualize_loss(history, "Training and Validation Loss", PATH_SAVE_MODEL)
    visualize_mdape(history, "Training and Validation mdape", PATH_SAVE_MODEL)

    model = keras.models.load_model(PATH_SAVE_MODEL+"my_checkpoint.h5", custom_objects={"mdape": mdape, 'loss_custom':loss_custom })
    print(model.evaluate(train_dataset))
    print(model.evaluate(val_dataset))

    dict_model = dict( 
        name = args.name_model,
        base_model = args.base_model,
        Args = dict (
            batch_size= args.batch_size,
            drop_out= args.drop_out,
            epochs= args.epochs,
            future= args.future,
            load_weight_imputer = args.load_weight_imputer,
            imputer_choose= args.imputer_choose,
            keep_col_percent= args.keep_col_percent,
            learning_rate= args.learning_rate,
            max_iter= args.max_iter,
            n_Dense= args.n_Dense,
            n_LSTM= args.n_LSTM,
            n_neighbors= args.n_neighbors,
            past= args.past,
            sequence_stride= args.sequence_stride,
            sequence_stride_val= args.sequence_stride_val,
            split_fraction= args.split_fraction,
            step= args.step,
            step_missing= args.step_missing,
            stop_early= args.stop_early,
            monitor= args.monitor,
            loss = args.loss,
            ),
        col = COL_NAME,
        min_x = X_min.tolist(),
        max_x = X_max.tolist(),
        min_y = Y_min.tolist(),
        max_y = Y_max.tolist(),
        path_model = PATH_SAVE_MODEL+"my_checkpoint.h5",
        path_list_imputer = list_imputer,
        path_list_imputer_out = list_imputer_out,
        path_save_model_yaml = PATH_SAVE_MODEL +'model_save.yml'
    )

    with open(PATH_SAVE_MODEL +'model_save.yml', 'w') as yaml_file:
        yaml.dump(dict_model, yaml_file, default_flow_style=False)

