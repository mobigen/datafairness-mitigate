# coding: utf-8

import numpy as np
import pandas as pd

from metric.dataset_metric import DatasetMetric

class ClassificationMetric(DatasetMetric):
    """Data와 Model 예측 결과로부터 예측 성능과 공정성에 관한 Metric을 제공"""
    def __init__(self, origin_df, prediction_df, protected_attribute_name, label_name):
        """
        Arguments:
            origin_df (pandas.DataFrame): 원본 데이터
            prediction_df (pandas.DataFrame): 레이블이 모델 예측값(Binary)로 치환된 모델 데이터
            protected_attribute_name (str): 원본, 모델 pandas.DataFrame 컬럼으로 정의된 Protected Attribute Name
            label_name (str): 원본, 모델 pandas.DataFrame 컬럼으로 정의된 Label Name
        """
        super(ClassificationMetric, self).__init__(origin_df, protected_attribute_name, label_name)

        self.pred_df = prediction_df

    def num_true_positive(self, privileged=None):
        """Model이 실제 Positive를 Positive라고 제대로 예측한 개체수"""
        mask_orig_pos = self.make_mask(self.orig_df, pos_neg='positive', privileged=privileged)
        mask_pred_pos = self.make_mask(self.pred_df, pos_neg='positive', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_pos, mask_pred_pos))

    def num_true_negative(self, privileged=None):
        """Model이 실제 Negative를 Negative라고 제대로 예측한 개체수"""
        mask_orig_neg = self.make_mask(self.orig_df, pos_neg='negative', privileged=privileged)
        mask_pred_neg = self.make_mask(self.pred_df, pos_neg='negative', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_neg, mask_pred_neg))

    def num_false_positive(self, privileged=None):
        """Model이 실제 Negative를 Positive라고 잘못 예측한 개체수"""
        mask_orig_neg = self.make_mask(self.orig_df, pos_neg='negative', privileged=privileged)
        mask_pred_pos = self.make_mask(self.pred_df, pos_neg='positive', privileged=privileged)
        return np.sum(np.logical_and(mask_orig_neg, mask_pred_pos))

    def num_false_negative(self, privileged=None):
        """Model이 실제 Positive를 Negative라고 잘못 예측한 개체수"""
        mask_orig_pos = self.make_mask(self.orig_df, pos_neg='positive', privileged=privileged)
        mask_pred_neg = self.make_mask(self.pred_df, pos_neg='negative', privileged=privileged)
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

    def selection_rate(self, privileged=None):
        r"""전체에서 Model이 예측한 Positive 개체가 차지하는 비율

        .. math:: Pr(\hat{Y}=1|D=d), \text{ d is one of } unprivileged, privileged \text{ or } all.
        """
        return self.num_pred_positive(privileged=privileged)/(self.num_positive(privileged=privileged)+
                                                              self.num_negative(privileged=privileged))

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

    def theil_index(self):
        r"""
        Generalized Entropy Index를 이용한 Theil Index 계산 [1]_.

        :math:`b_i = \hat{y}_i - y_i + 1` 일 때,

        .. math::

            \displaystyle\frac{1}{N} \sum_{i=1}^{N} \frac{b_i}{\mu} ln\Big(\frac{b_{i}}{\mu}\Big) =
            \displaystyle\frac{1}{N} \sum_{i=1}^{N} \frac{1}{\mu} ln\Bigg(\Big(\frac{b_{i}}{\mu}\Big)^{b_{i}}\Bigg)

        References:
            .. [1] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
               "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
               ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.

               https://arxiv.org/pdf/1807.00787
        """
        label_orig = self.orig_df[self.label_name].astype(np.float64)
        label_pred = self.pred_df[self.label_name].astype(np.float64)
        b = label_pred - label_orig + 1  # benefit score b
        mu_b = np.mean(b)

        alpha = 1
        if alpha == 1:
            # b_i=0 인 경우를 처리하기 위해, 로그의 성질에 의하여 b를 로그 내부 진수의 지수 자리로 이동
            # np.mean(np.log(b/mu_b)*(b/mu_b)) 이면, b_i=0일때 계산할 수 없음
            return np.mean(np.log((b/mu_b)**b)/mu_b)
        else:
            raise NotImplemented('It\'s not needed that alpha not 1.')
