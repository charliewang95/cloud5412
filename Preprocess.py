import csv
import random
import pickle
import pandas as pd
import numpy as np
import config
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def read_raw(csvfile = 'ANSC_Data.csv'):
    '''
    Generate pandas dataframe from csv file
    Probably no need to use if getting data from COSMOS DB
    '''
    all_data = pd.read_csv(csvfile)
    return all_data

def modify_csv():
    '''
    Change csv file to train data with three consecutive periods
    Only use the first time
    '''
    feature_list = config.feature_list_diff
    feature_list2 = config.feature_list_same+['newvar']

    all_data = []
    with open('ANSC_Data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            all_data.append(row)

    new_all_data = []
    for idx in range(len(all_data)-2):
        if all_data[idx]['ID'] == all_data[idx+1]['ID'] and all_data[idx]['ID'] == all_data[idx+2]['ID']:
            new_dict = {}
            for feature in feature_list:
                new_dict[feature+'0'] = all_data[idx][feature]
                new_dict[feature+'1'] = all_data[idx+1][feature]
                new_dict[feature+'2'] = all_data[idx+2][feature]
            for feature in feature_list2:
                new_dict[feature] = all_data[idx][feature]
            new_all_data.append(new_dict)

    csv_columns = config.feature_list_all+['newvar']

    with open('ANSC_trio_Data.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for data in new_all_data:
            writer.writerow(data)

def preprocess(df):
    '''
    All the preprocess needed to be done: For Calving time
    '''
    #Change newvar to Boolean Variable: Calving in 6 hrs
    df['newvar'] = df['newvar'] < 3
    #Deal with missing value, currently use mean
    df.fillna(df.mean(), inplace=True)
    return df


def train_dev_ease(df, data_path = None, dev_size = 0.2):
    '''
    Perform train dev split for training EASE model (Predict calving difficulty)
        df: The dataframe
        data_path: The directory to store splitted datasets
        dev_size: The proportion of validation data
    Includes feature selection and splitting into training/validation set
    '''
    # A List of training features
    feature_list = config.feature_list_ease
    X = df[feature_list]
    y = df['EASE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dev_size)
    if data_path is not None:
        X_train.to_pickle(data_path+'/X_train.pkl')
        X_test.to_pickle(data_path+'/X_test.pkl')
        y_train.to_pickle(data_path+'/y_train.pkl')
        y_test.to_pickle(data_path+'/y_test.pkl')
    return X_train, X_test, y_train, y_test


def train_dev(df, data_path = None, dev_size = 0.2):
    '''
    Perform train dev split for training calving time model (Predict calving time)
        df: The dataframe
        data_path: The directory to store splitted datasets
        dev_size: The proportion of validation data
    Includes feature selection and splitting into training/validation set
    '''
    feature_list = config.feature_list_all
    X = df[feature_list]
    y = df['newvar']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dev_size)
    if data_path is not None:
        X_train.to_pickle(data_path+'/X_train.pkl')
        X_test.to_pickle(data_path+'/X_test.pkl')
        y_train.to_pickle(data_path+'/y_train.pkl')
        y_test.to_pickle(data_path+'/y_test.pkl')
    return X_train, X_test, y_train, y_test
