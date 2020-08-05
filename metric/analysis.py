# coding: utf-8

import numpy as np
import pandas as pd

class ClassificationMetric:
    """Data와 Model 예측 결과로부터 예측 성능과 공정성에 관한 Metric을 제공"""
    def __init__(self, origin_df, prediction_df, protected_attribute_name, label_name):
        """
        Arguments:
            origin_df (pandas.DataFrame): 원본 데이터
            prediction_df (pandas.DataFrame): 레이블이 모델 예측값(Binary)로 치환된 모델 데이터
            protected_attribute_name (str): 원본, 모델 pandas.DataFrame 컬럼으로 정의된 Protected Attribute Name
            label_name (str): 원본, 모델 pandas.DataFrame 컬럼으로 정의된 Label Name
        """
        self.orig_df = origin_df
        self.pred_df = prediction_df

        self.prot_name = protected_attribute_name
        self.label_name = label_name

    def make_mask(self, df_type, pos_neg, privileged=None):
        """특정 조건에 따라 Metric에 사용할 Mask를 생성

        Arguments:
            df_type (str):
                origin 또는 prediction dataset 중 한 종류를 mask 한다
            pos_neg (str):
                positive 또는 negative 중 한 종류를 mask 한다
            privileged (bool or None):
                None: 전체 Group에 대하여 mask 한다
                True: Protected attribute 값이 1인 Group 만 mask 한다
                False: Protected attribute 값이 0인 Group 만 mask 한다
        """
        if df_type == 'origin':
            df_pos_mask = (self.orig_df[self.label_name] == True)
        elif df_type == 'prediction':
            df_pos_mask = (self.pred_df[self.label_name] == True)
        else:
            raise ValueError('Argument \'df_type\' must be \'origin\' or \'prediction\'')

        if pos_neg == 'positive':
            mask = df_pos_mask
        elif pos_neg == 'negative':
            mask = ~df_pos_mask
        else:
            raise ValueError('Argument \'pos_neg\' must be \'positive\' or \'negative\'')

        if privileged is not None:
            if privileged:
                mask = np.logical_and(mask, self.orig_df[self.prot_name] == 1)
            else:
                mask = np.logical_and(mask, self.orig_df[self.prot_name] == 0)
        return mask

    def num_positive(self, privileged=None):
        """Positive 개체수를 카운트하여 반환"""
        return np.sum(self.make_mask(df_type='origin', pos_neg='positive', privileged=privileged))

    def num_negative(self, privileged=None):
        """Negative 개체수를 카운트하여 반환"""
        return np.sum(self.make_mask(df_type='origin', pos_neg='negative', privileged=privileged))

    def num_true_positive(self, privileged=None):
        """Model이 실제 Positive를 Positive라고 제대로 예측한 개체수"""
        mask_orig_pos = self.make_mask(df_type='origin', pos_neg='positive', privileged=privileged)
        mask_pred_pos = self.make_mask(df_type='prediction', pos_neg='positive', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_pos, mask_pred_pos))

    def num_true_negative(self, privileged=None):
        """Model이 실제 Negative를 Negative라고 제대로 예측한 개체수"""
        mask_orig_neg = self.make_mask(df_type='origin', pos_neg='negative', privileged=privileged)
        mask_pred_neg = self.make_mask(df_type='prediction', pos_neg='negative', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_neg, mask_pred_neg))

    def num_false_positive(self, privileged=None):
        """Model이 실제 Negative를 Positive라고 잘못 예측한 개체수"""
        mask_orig_neg = self.make_mask(df_type='origin', pos_neg='negative', privileged=privileged)
        mask_pred_pos = self.make_mask(df_type='prediction', pos_neg='positive', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_neg, mask_pred_pos))

    def num_false_negative(self, privileged=None):
        """Model이 실제 Positive를 Negative라고 잘못 예측한 개체수"""
        mask_orig_pos = self.make_mask(df_type='origin', pos_neg='positive', privileged=privileged)
        mask_pred_neg = self.make_mask(df_type='prediction', pos_neg='negative', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_pos, mask_pred_neg))

    def num_pred_positive(self, privileged=None):
        """Model이 예측한 총 Positive 개체수"""
        return self.num_true_positive(privileged=privileged) + self.num_false_positive(privileged=privileged)

    def num_pred_negative(self, privileged=None):
        """Model이 예측한 총 Negative 개체수"""
        return self.num_true_negative(privileged=privileged) + self.num_false_negative(privileged=privileged)

    def perform_confusion_matrix(self, privileged=None):
        """Confusion Matrix 계산"""
        P = self.num_positive(privileged=privileged)
        N = self.num_negative(privileged=privileged)
        TP = self.num_true_positive(privileged=privileged)
        TN = self.num_true_negative(privileged=privileged)
        FP = self.num_false_positive(privileged=privileged)
        FN = self.num_false_negative(privileged=privileged)
        return dict(TPR=TP/P, TNR=TN/N, FPR=FP/N, FNR=FN/P,
                    PPV=TP/(TP+FP), NPV=TN/(TN+FN), FDR=FP/(TP+FP), FOR=FN/(TN+FN),
                    ACC=(TP+TN)/(P+N) if P+N > 0. else np.float64(0.))

    def accuracy(self, privileged=None):
        r"""Accuracy 계산

        .. math:: \frac{(TP + TN)}{(P + N)}
        """
        P = self.num_positive(privileged=privileged)
        N = self.num_negative(privileged=privileged)
        TP = self.num_true_positive(privileged=privileged)
        TN = self.num_true_negative(privileged=privileged)
        return (TP+TN)/(P+N)

    def balanced_accuracy(self, privileged=None):
        r"""Balanced Accuary 계산

        .. math:: 0.5 \times Pr(\hat{Y}|Y=1) + Pr(\hat{Y}=0|Y=0)
        """
        return 0.5 * (self.true_positive_rate(privileged=privileged) +
                      self.true_negative_rate(privileged=privileged))

    def true_positive_rate(self, privileged=None):
        r"""실제 Positive에서 True Positive가 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=1|Y=1)}{Pr(Y=1)}
        """
        return self.num_true_positive(privileged=privileged)/self.num_positive(privileged=privileged)

    def true_negative_rate(self, privileged=None):
        r"""실제 Negative에서 True Negative가 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=0|Y=0)}{Pr(Y=0)}
        """
        return self.num_true_negative(privileged=privileged)/ self.num_negative(privileged=privileged)

    def false_positive_rate(self, privileged=None):
        r"""실제 Negative에서 False Positive가 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=1|Y=0)}{Pr(Y=0)}
        """
        return self.num_false_positive(privileged=privileged)/self.num_negative(privileged=privileged)

    def false_negative_rate(self, privileged=None):
        r"""실제 Positive에서 False Negative가 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=0|Y=1)}{Pr(Y=1)}
        """
        return self.num_false_negative(privileged=privileged)/self.num_positive(privileged=privileged)

    def positive_predictive_value(self, privileged=None):
        r"""Model이 예측한 Positive에서 실제 Positive가 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=1|Y=1)}{Pr(\hat{Y}=1)}
        """
        return self.num_true_positive(privileged=privileged)/self.num_pred_positive(privileged=privileged)

    def negative_predictive_value(self, privileged=None):
        r"""Model이 예측한 Negative에서 실제 Negative가 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=0|Y=0)}{Pr(\hat{Y}=0)}
        """
        return self.num_true_negative(privileged=privileged)/self.num_pred_negative(privileged=privileged)

    def false_discovery_rate(self, privileged=None):
        r"""Model이 예측한 Positive에서 Positive로 잘못 예측한(실제 Negative인) 개체들이 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=1|Y=0)}{Pr(\hat{Y}=1)}
        """
        return self.num_false_positive(privileged=privileged)/self.num_pred_positive(privileged=privileged)

    def false_omission_rate(self, privileged=None):
        r"""Model이 예측한 Negative에서 Negative로 잘못 예측한(실제 Positive인) 개체들이 차지하는 비율

        .. math:: \frac{Pr(\hat{Y}=0|Y=1)}{Pr(\hat{Y}=0)}
        """
        return self.num_false_negative(privileged=privileged)/self.num_pred_negative(privileged=privileged)

    def sensitivity(self, privileged=None):
        """True Positive Rate의 Alias"""
        return self.true_positive_rate(privileged=privileged)

    def recall(self, privileged=None):
        """True Positive Rate의 Alias"""
        return self.true_positive_rate(privileged=privileged)

    def specificity(self, privileged=None):
        """True Negative Rate의 Alias"""
        return self.true_negative_rate(privileged=privileged)

    def fall_out(self, privileged=None):
        """False Positive Rate의 Alias"""
        return self.false_positive_rate(privileged=privileged)

    def miss_rate(self, privileged=None):
        """False Negative Rate의 Alias"""
        return self.false_negative_rate(privileged=privileged)

    def num_instances(self, privileged=None):
        if privileged is None:
            return self.orig_df.shape[0]
        elif privileged:
            return self.orig_df[self.orig_df[self.prot_name] == 1].shape[0]
        elif not privileged:
            return self.orig_df[self.orig_df[self.prot_name] == 0].shape[0]
        else:
            raise ValueError('Invalid Argument \'privileged\': {}'.format(privileged))

    def base_rate(self, privileged=None):
        r"""전체에서 Positive 개체가 차지하는 비율

        .. math:: Pr(Y=1)
        """
        return self.num_positive(privileged=privileged)/(self.num_positive(privileged=privileged)+
                                                         self.num_negative(privileged=privileged))

    def selection_rate(self, privileged=None):
        r"""전체에서 Model이 예측한 Positive 개체가 차지하는 비율

        .. math:: Pr(\hat{Y}=1|D=d), \text{ d is one of } unprivileged, privileged \text{ or } all.
        """
        return self.num_pred_positive(privileged=privileged)/(self.num_positive(privileged=privileged)+
                                                              self.num_negative(privileged=privileged))

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
        .. math:: \frac{Pr(\hat{Y}=1|D=unprivileged)}{Pr(\hat{Y}=1|D=privileged)}
        """
        return self.fairness_ratio(self.selection_rate)

    def statistical_parity_difference(self):
        r"""
        .. math:: Pr(\hat{Y}=1|D=Unprivileged) - Pr(\hat{Y}=1|D=privileged)
        """
        return self.fairness_difference(self.selection_rate)

    def mean_difference(self):
        """statistical_parity_difference의 Alias"""
        return self.statistical_parity_difference()

    def average_odds_difference(self):
        r"""
        .. math::
            0.5 \times \left\{\Big((Pr(\hat{Y}=1|Y=0, D=unprivileged) - Pr(\hat{Y}=1|Y=0, D=privileged)\Big) +

            \Big(Pr(\hat{Y}=1|Y=1, D=unprivileged) - Pr(\hat{Y}=1|Y=1, D=privileged)\Big)\right\}
        """
        return 0.5 * (self.fairness_difference(self.false_positive_rate) +
                      self.fairness_difference(self.true_positive_rate))

    def equal_opportunity_difference(self):
        r"""
        .. math:: Pr(\hat{Y}=1|Y=1, D=unprivileged) - Pr(\hat{Y}=1|Y=1, D=privileged)
        """
        return self.fairness_difference(self.true_positive_rate)