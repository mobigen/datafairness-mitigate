# coding: utf-8

import numpy as np
import pandas as pd

class DatasetMetric:
    def __init__(self, origin_df, protected_attribute_name, label_name):
        """
        Arguments:
            origin_df (pandas.DataFrame): 원본 데이터
            protected_attribute_name (str): 원본, 모델 pandas.DataFrame 컬럼으로 정의된 Protected Attribute Name
            label_name (str): 원본, 모델 pandas.DataFrame 컬럼으로 정의된 Label Name
        """
        self.orig_df = origin_df

        self.prot_name = protected_attribute_name
        self.label_name = label_name

    def make_mask(self, df, pos_neg, privileged=None):
        """특정 조건에 따라 Metric에 사용할 Mask를 생성

        Arguments:
            df (pandas.DataFrame):
                mask 할 origin 또는 prediction dataset 중 한 종류
            pos_neg (str):
                positive 또는 negative 중 한 종류를 mask 한다
            privileged (bool or None):
                None: 전체 Group에 대하여 mask 한다
                True: Protected attribute 값이 1인 Group 만 mask 한다
                False: Protected attribute 값이 0인 Group 만 mask 한다
        """
        df_pos_mask = (df[self.label_name] == True)

        if pos_neg == 'positive':
            mask = df_pos_mask
        elif pos_neg == 'negative':
            mask = ~df_pos_mask
        else:
            raise ValueError('Argument \'pos_neg\' must be \'positive\' or \'negative\'')

        if privileged is not None:
            if privileged:
                mask = np.logical_and(mask, df[self.prot_name] == 1)
            else:
                mask = np.logical_and(mask, df[self.prot_name] == 0)
        return mask

    def num_instances(self, privileged=None):
        if privileged is None:
            return self.orig_df.shape[0]
        elif privileged:
            return self.orig_df[self.orig_df[self.prot_name] == 1].shape[0]
        elif not privileged:
            return self.orig_df[self.orig_df[self.prot_name] == 0].shape[0]
        else:
            raise ValueError('Invalid Argument \'privileged\': {}'.format(privileged))

    def num_positive(self, privileged=None):
        """Positive 개체수를 카운트하여 반환"""
        return np.sum(self.make_mask(self.orig_df, pos_neg='positive', privileged=privileged))

    def num_negative(self, privileged=None):
        """Negative 개체수를 카운트하여 반환"""
        return np.sum(self.make_mask(self.orig_df, pos_neg='negative', privileged=privileged))

    def base_rate(self, privileged=None):
        r"""전체에서 Positive 개체가 차지하는 비율

        .. math:: Pr(Y=1)
        """
        return self.num_positive(privileged=privileged)/self.num_instances(privileged=privileged)

    def fairness_ratio(self, fn):
        r"""Unprivileged Group 과 Privileged Group 간 어떤 함수 f를 적용했을때 그 결과의 비율

        .. math:: \frac{f(D=unprivileged)}{f(D=privileged)}
        """
        return fn(privileged=False) / fn(privileged=True)

    def fairness_difference(self, fn):
        r"""Unprivileged Group 과 Privileged Group 간 어떤 함수 f를 적용했을때 그 결과의 차

        .. math:: f(D=unprivileged) - f(D=privileged)
        """
        return fn(privileged=False) - fn(privileged=True)

    def disparate_impact(self):
        r"""
        .. math:: \frac{Pr(Y=1|D=unprivileged)}{Pr(Y=1|D=privileged)}
        """
        return self.fairness_ratio(self.base_rate)

    def statistical_parity_difference(self):
        r"""
        .. math:: Pr(Y=1|D=Unprivileged) - Pr(Y=1|D=privileged)
        """
        return self.fairness_difference(self.base_rate)

    def mean_difference(self):
        """statistical_parity_difference의 Alias"""
        return self.statistical_parity_difference()
