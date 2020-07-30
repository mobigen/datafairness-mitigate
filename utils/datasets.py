# coding: utf-8

import os

import numpy as np
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_adults_df():
    train_path = os.path.join(root_dir, 'data', 'raw', 'adult', 'adult.data')
    test_path = os.path.join(root_dir, 'data', 'raw', 'adult', 'adult.test')

    na_values = '?'
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']

    train = pd.read_csv(train_path, header=None, names=column_names,
                    skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
        skipinitialspace=True, na_values=na_values)

    return pd.concat([train, test], ignore_index=True)

def convert_label_to_binary(df, label_name, favorable_label_classes):
    """레이블을 특정 이진값 또는 0,1 이진값으로 변환"""
    favorable_converted_label = 1.
    unfavorable_converted_label = 0.
    if callable(favorable_label_classes):
        # favorable_label_classes True, False를 리턴하는 함수일때
        df[label_name] = df[label_name].apply(favorable_label_classes)
    elif np.issubdtype(df[label_name], np.number) and len(set(df[label_name])) == 2:
        # labels are already binary; don't change them
        favorable_converted_label = favorable_label_classes[0]
        unfavorable_converted_label = set(df[label_name]).difference(favorable_label_classes).pop()
    else:
        # find all instances which match any of the favorable classes
        pos = np.logical_or.reduce(np.equal.outer(favorable_label_classes, df[label_name].to_numpy()))
        df.loc[pos, label_name] = favorable_converted_label
        df.loc[~pos, label_name] = unfavorable_converted_label

def preprocess_df(df, label_name, favorable_label_classes, custom_preproc=None):
    if custom_preproc: df = custom_preproc(df)
    convert_label_to_binary(df, label_name, favorable_label_classes)