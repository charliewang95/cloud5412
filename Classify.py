import csv
import random
import pickle
import pandas as pd
import numpy as np
from Preprocess import *
import config
import requests
import json
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def random_forest_train(data_path = None, model_path = 'Model/for_test', model_type = 2):
    '''
    Train a random forest model and store it for future use
    '''
    #Create new dataframes for train/dev and save it
    if data_path is None:
        dataframe = read_raw('ANSC_trio_Data.csv')
        dataframe = preprocess(dataframe)
        X_train, X_test, y_train, y_test = train_dev(dataframe, 'Data/trio_2')
    #Read processed dataset from files
    else:
        X_train = pd.read_pickle(data_path+'/X_train.pkl')
        X_test = pd.read_pickle(data_path+'/X_test.pkl')
        y_train = pd.read_pickle(data_path+'/y_train.pkl')
        y_test = pd.read_pickle(data_path+'/y_test.pkl')
    if model_type == 1:
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', class_weight = 'balanced')
    else:
        clf = GradientBoostingClassifier(n_estimators=2000)
    clf.fit(X_train, y_train)
    pickle.dump(clf, open(model_path, 'wb'))
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    evaluate(y_pred, y_test)

def random_forest_train_ease(data_path = None, model_path = 'Model/ease_2'):
    '''
    Train a random forest model and store it for future use
    '''
    #Create new dataframes for train/dev and save it
    if data_path is None:
        dataframe = read_raw('ANSC_Data.csv')
        dataframe = preprocess(dataframe)
        X_train, X_test, y_train, y_test = train_dev_ease(dataframe, 'Data/ease_2')
    #Read processed dataset from files
    else:
        X_train = pd.read_pickle(data_path+'/X_train.pkl')
        X_test = pd.read_pickle(data_path+'/X_test.pkl')
        y_train = pd.read_pickle(data_path+'/y_train.pkl')
        y_test = pd.read_pickle(data_path+'/y_test.pkl')
    clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', class_weight = 'balanced')
    clf.fit(X_train, y_train)
    pickle.dump(clf, open(model_path, 'wb'))
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def predict(model_path, data_path):
    '''
    Use the calving time model to predict the whether a cow is calving in next 6 hrs.
        data_path: path to testing data and labels
        model_path: path to the classifier model
    '''
    X_test = pd.read_pickle(data_path+'/X_test.pkl')
    y_test = pd.read_pickle(data_path+'/y_test.pkl')
    clf = pickle.load(open(model_path, 'rb'))
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    evaluate(y_pred, y_test)


def predict_ease(model_path, data_path):
    '''
    Use the ease model to predict the calving difficulty
        data_path: path to testing data and labels
        model_path: path to the classifier model
    '''
    X_test = pd.read_pickle(data_path+'/X_test.pkl')
    y_test = pd.read_pickle(data_path+'/y_test.pkl')
    clf = pickle.load(open(model_path, 'rb'))
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def evaluate(y_pred, y_test):
    '''
    This function shows detail of calculating F1 score on model
    Not really needed in the final cloud system, just for model eval/selection
    '''
    y_test = y_test = np.array(y_test)
    TP,FP,FN,TN = 0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]==1:
            TP += 1
        elif y_pred[i]==1 and y_test[i]!=y_pred[i]:
            FP += 1
        elif y_test[i]==y_pred[i]==0:
            TN += 1
        else:
            FN += 1
    precision = float(TP)/float(TP+FP)
    recall = float(TP)/float(TP+FN)
    F1 = (2*precision*recall)/(precision+recall)
    print("True Positives: ",TP)
    print("False Positives: ",FP)
    print("False Negatives: ",FN)
    print("True Negatives: ",TN)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", F1)

def Predict_from_ID(cowID, model_path='Model/trio_3'):

    #Issue request to get json data
    url = 'https://cow-rumination-data-app1.azurewebsites.net/'+str(cowID)
    print(url)
    r = requests.get(url = url)
    json_dict = json.loads(r.text)
    new_dict = {}

    #Parse json as a dictionary file, handle missing values
    for feat in config.feature_list_same:
        if feat not in json_dict:
            if feat not in json_dict['entries'][0]:
                new_dict[feat] = config.mean_feat[feat]
            else:
                new_dict[feat] = json_dict['entries'][0][feat]
        else:
            new_dict[feat] = json_dict[feat]

    for feat in config.feature_list_diff:
        for time in range(3):
            if feat not in json_dict['entries'][time]:
                new_dict[feat+str(time)] = config.mean_feat[feat+str(time)]
            else:
                new_dict[feat+str(time)] = json_dict['entries'][time][feat]
    
    df = pd.DataFrame(new_dict, index = ['ID'])
    clf = pickle.load(open(model_path, 'rb'))
    
    y_pred=clf.predict(df[config.feature_list_all])
    print(y_pred)
    return y_pred[0]

def Ease_from_ID(cowID, model_path='Model/ease_2'):
    #Issue request to get json data
    url = 'https://cow-rumination-data-app1.azurewebsites.net/'+str(cowID)
    print(url)
    r = requests.get(url = url)
    json_dict = json.loads(r.text)

    #Choose the five most recent entries (Or all entries if <=5 totals)
    evals = min(len(json_dict['entries']), 5)
    clf = pickle.load(open(model_path, 'rb'))
    result_list = []
    for i in range(evals):
        new_dict = {}
        for feat in config.feature_list_ease:
            if feat in json_dict:
                new_dict[feat] = json_dict[feat]
            elif feat not in json_dict['entries'][0]:
                new_dict[feat] = config.mean_feat[feat+str(0)]
            else:
                new_dict[feat] = json_dict['entries'][0][feat]
        df = pd.DataFrame(new_dict, index = ['ID'])
        y_pred=clf.predict(df[config.feature_list_ease])
        result_list.append(y_pred[0])

    print(result_list)
    print(max(set(result_list), key=result_list.count))
    return max(set(result_list), key=result_list.count)

#Ease_from_ID(sys.argv[1])
#Predict_from_ID(sys.argv[1])
#predict('Model/trio_3', 'Data/trio_2')
#predict_ease('Model/ease_1','Data/ease_1')
random_forest_train()#data_path = 'Data/ease_1')
