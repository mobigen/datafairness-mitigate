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

def handle_missing(df, handle_funcs=None):
    """결측치 처리"""
    if handle_funcs: raise NotImplementedError
    else:
        dropped_df = df.dropna()
        drop_cnt = df.shape[0] - dropped_df.shape[0]
        print(f"Missing Data: {drop_cnt} rows removed.")
        df = dropped_df
    return df

def convert_one_hot_features(df, column_names=None):
    """Categorical Column을 One-Hot으로 변환"""
    if column_names:
        df = pd.get_dummies(df, prefix_sep='=', columns=column_names)
    return df

def convert_protected_attribute_to_binary(df, protected_attribute_names, privileged_classes):
    privileged_converted_values = [1.]
    unprivileged_converted_values = [0.]
    for attr, vals in zip(protected_attribute_names, privileged_classes):
        if callable(vals):
            # vals가 True, False를 리턴하는 함수일때
            df[attr] = df[attr].apply(vals)
        elif np.issubdtype(df[attr].dtype, np.number):
            # this attribute is numeric; no remapping needed
            privileged_converted_values = vals
            unprivileged_converted_values = list(set(df[attr]).difference(vals))
        else:
            # find all instances which match any of the attribute values
            priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
            df.loc[priv, attr] = privileged_converted_values[0]
            df.loc[~priv, attr] = unprivileged_converted_values[0]
    return df

def convert_label_to_binary(df, label_name, favorable_label_classes):
    """레이블을 특정 이진값 또는 0,1 이진값으로 변환"""
    favorable_converted_label = 1.  # favorable label class는 1
    unfavorable_converted_label = 0.  # unfavorable label class는 0
    if callable(favorable_label_classes):
        # favorable_label_classes가 True, False를 리턴하는 함수일때
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
    return df

def preprocess_df(df,
                  protected_attribute_names, privileged_class,
                  label_name, favorable_label_classes,
                  custom_preproc=None, one_hot_column_names=None):
    if custom_preproc: df = custom_preproc(df)
    df = handle_missing(df)
    df = convert_one_hot_features(df, column_names=one_hot_column_names)
    df = convert_protected_attribute_to_binary(df, protected_attribute_names, privileged_class)
    df = convert_label_to_binary(df, label_name, favorable_label_classes)
    return df

def describe_df(df, detail=False):
    print('Shape:', df.shape)
    if detail:
        pd.set_option('display.max_rows', len(df))
        pd.set_option('display.max_columns', None)
        print(df.describe())
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')

def split_df(df, shuffle=True, ratio=0.7):
    nrow = df.index.values.shape[0]
    train_nrow = int(nrow * ratio)

    mask = np.zeros(nrow, dtype=np.bool)
    if not shuffle:
        mask[:train_nrow] = True
    else:
        train_ind = np.random.choice(nrow, size=train_nrow, replace=False)
        mask[train_ind] = True

    return df[mask], df[~mask]
