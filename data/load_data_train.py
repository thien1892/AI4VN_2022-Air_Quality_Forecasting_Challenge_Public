import pandas as pd
import os
from tqdm import tqdm

def load_data_train(path_location, path_folder):
    """
    Load data train input (In example: data of 11 station)
    Args:
        path_location: dir path location file --> read satation order
        path_folder: dir path of folder contain data train input or data train output
    Return:
        file csv data combine train input or output
    """
    df_total_train = pd.DataFrame()
    for i in tqdm(pd.read_csv(path_location)['station'].map(lambda x: str(x)+ '.csv').to_list()):
        try:
            df = pd.read_csv(os.path.join(path_folder, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            dict_col_change = {
            'PM2.5': 'PM2.5_' + name_location_i,
            'humidity': 'humidity_'+ name_location_i,
            'temperature': 'temperature_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_total_train.columns) < 2:
                df_total_train['timestamp'] = df['timestamp']
            df_total_train = df_total_train.merge(df, on = 'timestamp')
        except:
            print(f'No found file: {i} !')
    df_total_train['timestamp'] = pd.to_datetime(df_total_train['timestamp'], dayfirst= True)
    df_total_train = df_total_train.set_index('timestamp')

    return df_total_train

def check_feature(df, p):
    """
    Keep column of data df has missing data percent < p%
    Args:
        df: DataFrame
        p: percent
    Return:
        List columns have data minsing/ total < p%
    """
    k = 0
    list_feature = []
    for i in df.columns:
        n_miss = df[[i]].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        if perc.values[0]< p:
            # print('> %s, Missing: %d (%.1f%%)' % (i, n_miss, perc))
            k += 1
            list_feature.append(i)
    print(f'feature < {p}%: {k} / total: {len(df.columns)}')
    return list_feature

def normalize(data, train_split):
    # Normalize Min-Max data
    data_min = data[:train_split].min(axis=0)
    data_max = data[:train_split].max(axis=0)
    return (data - data_min) / (data_max - data_min) , data_min, data_max

def load_data_test(path_folder, col_name):
    """
    Load data test to predict and submit
    Args:
        path_folder: dir path of data test
        col_name(list): list columns order by data train
    Return:
        dict data of each folder in path_folder test.
    """
    dict_data_submit = {}
    for k in tqdm(os.listdir(path_folder)):
        PUBLIC_TEST_INPUT_PATH_FOLDER = os.path.join(path_folder, k)
        df_concat = pd.DataFrame()
        for i in os.listdir(PUBLIC_TEST_INPUT_PATH_FOLDER):
            df = pd.read_csv(os.path.join(PUBLIC_TEST_INPUT_PATH_FOLDER, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            dict_col_change = {
            'PM2.5': 'PM2.5_' + name_location_i,
            'humidity': 'humidity_'+ name_location_i,
            'temperature': 'temperature_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_concat.columns) < 2:
                df_concat['timestamp'] = df['timestamp']
            df_concat = df_concat.merge(df, on = 'timestamp')
        value_k = df_concat[col_name].values
        dict_data_submit[k] = value_k
    return dict_data_submit


def load_data_public_test(path_folder, col_name):
    # Return a DataFrame public test
    df_total_public_test = pd.DataFrame()
    for k in tqdm(os.listdir(path_folder)):
        PUBLIC_TEST_INPUT_PATH_FOLDER = os.path.join(path_folder, k)
        df_concat = pd.DataFrame()
        for i in os.listdir(PUBLIC_TEST_INPUT_PATH_FOLDER):
            df = pd.read_csv(os.path.join(PUBLIC_TEST_INPUT_PATH_FOLDER, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            dict_col_change = {
            'PM2.5': 'PM2.5_' + name_location_i,
            'humidity': 'humidity_'+ name_location_i,
            'temperature': 'temperature_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_concat.columns) < 2:
                df_concat['timestamp'] = df['timestamp']
            df_concat = df_concat.merge(df, on = 'timestamp')
        # df_concat = df_concat[['timestamp'] + COL_NAME]
        df_concat = df_concat[['timestamp'] + col_name]
        # df_concat['folder public test'] = k
        if len(df_total_public_test.columns) < 2:
            df_total_public_test = pd.DataFrame(columns= df_concat.columns)
        df_total_public_test = pd.concat((df_total_public_test, df_concat))

    df_total_public_test['timestamp'] = pd.to_datetime(df_total_public_test['timestamp'], dayfirst = True)
    df_total_public_test = df_total_public_test.drop_duplicates(subset=['timestamp'])
    df_total_public_test = df_total_public_test.set_index('timestamp')

    return df_total_public_test