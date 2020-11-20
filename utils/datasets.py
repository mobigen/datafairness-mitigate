# coding: utf-8

import os

import numpy as np
import pandas as pd

from utils import download_data

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_adults_df():
    train_path = os.path.join(root_dir, 'data', 'raw', 'adult', 'adult.data')
    test_path = os.path.join(root_dir, 'data', 'raw', 'adult', 'adult.test')

    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        print('데이터 파일이 없습니다. 다운로드 합니다.')
        download_data.download('adult')

    na_values = '?'
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']

    train = pd.read_csv(train_path, header=None, names=column_names,
                    skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
        skipinitialspace=True, na_values=na_values)

    df = pd.concat([train, test], ignore_index=True)

    # specific preprocess
    def group_edu(x):
        if x <= 5: return '<6'
        elif x >= 13: return '>12'
        else: return x

    def age_cut(x):
        if x >= 70: return '>=70'
        else: return x

    def group_race(x):
        if x == 'White': return 1.
        else: return 0.

    df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
    df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))
    df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
    df['Education Years'] = df['Education Years'].astype('category')
    df['race'] = df['race'].apply(lambda x: group_race(x))
    df['sex'] = df['sex'].replace({'Female': 0., 'Male': 1.})

    # select feature
    features_to_keep = {'Age (decade)', 'Education Years', 'sex', 'race', 'income-per-year'}
    df = df[sorted(features_to_keep, key=df.columns.get_loc)]
    return df

def get_bank_df():
    data_path = os.path.join(root_dir, 'data', 'raw', 'bank', 'bank-additional-full.csv')

    if not os.path.isfile(data_path):
        print('데이터 파일이 없습니다. 다운로드 합니다.')
        download_data.download('bank')

    df = pd.read_csv(data_path, sep=';', na_values='unknown')
    return df
