# coding: utf-8

import numpy as np
import pandas as pd

class ClassificationMetric:
    def __init__(self, origin_df, prediction_df, protected_attribute_name, label_name):
        self.orig_df = origin_df
        self.pred_df = prediction_df

        self.prot_name = protected_attribute_name
        self.label_name = label_name

        self.origin_pos_mask = None

    def make_mask(self, privileged=None):
        """특정 조건에 따라 Metric에 사용할 Mask를 생성
        Arguments:
        - privileged: (None/True/False)
            - None: 전체 Group에 대하여 계산
            - True: Protected attribute 값이 1인 Group에 대하여 계산
            - False: Pretected attribute 값이 0인 Group에 대하여 계산"""
        if self.origin_pos_mask is None:
            self.origin_pos_mask = (self.orig_df[self.label_name] == True)

        mask = self.origin_pos_mask
        if privileged is not None:
            if privileged:
                mask = np.logical_and(self.origin_pos_mask, self.orig_df[self.prot_name] == 1)
            else:
                mask = np.logical_and(self.origin_pos_mask, self.orig_df[self.prot_name] == 0)
        return mask

    def num_positive(self, privileged=None):
        """Positive 개체수를 카운트하여 반환."""
        return np.sum(self.make_mask(privileged=privileged))

    def num_negative(self, privileged=None):
        return np.sum(~self.make_mask(privileged=privileged))

    def num_true_positive(self, privileged=None):
        return np.sum(
            (self.orig_df[self.label_name][self.make_mask(privileged=privileged)] ==
             self.pred_df[self.label_name][self.make_mask(privileged=privileged)]))

    def num_true_negative(self, privileged=None):
        return np.sum(
            (self.orig_df[self.label_name][~self.make_mask(privileged=privileged)] ==
             self.pred_df[self.label_name][~self.make_mask(privileged=privileged)]))

    def num_false_positive(self, privileged=None):
        return np.sum(
            (self.orig_df[self.label_name][~self.make_mask(privileged=privileged)] !=
             self.pred_df[self.label_name][~self.make_mask(privileged=privileged)]))

    def num_false_negative(self, privileged=None):
        return np.sum(
            (self.orig_df[self.label_name][self.make_mask(privileged=privileged)] !=
             self.pred_df[self.label_name][self.make_mask(privileged=privileged)]))

    def num_pred_positive(self, privileged=None):
        return self.num_true_positive(privileged=privileged) + self.num_false_positive(privileged=privileged)

    def num_pred_negative(self, privileged=None):
        return self.num_true_negative(privileged=privileged) + self.num_false_negative(privileged=privileged)

    def perform_confusion_matrix(self, privileged=None):
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
        P = self.num_positive(privileged=privileged)
        N = self.num_negative(privileged=privileged)
        TP = self.num_true_positive(privileged=privileged)
        TN = self.num_true_negative(privileged=privileged)
        return (TP+TN)/(P+N)

    def true_positive_rate(self, privileged=None):
        return self.num_positive(privileged=privileged)/self.num_true_positive(privileged=privileged)

    def true_negative_rate(self, privileged=None):
        return self.num_negative(privileged=privileged)/self.num_true_negative(privileged=privileged)

    def false_positive_rate(self, privileged=None):
        return self.num_false_positive(privileged=privileged)/self.num_negative(privileged=privileged)

    def false_negative_rate(self, privileged=None):
        return self.num_false_negative(privileged=privileged)/self.num_positive(privileged=privileged)

    def positive_predictive_value(self, privileged=None):
        return self.num_true_positive(privileged=privileged)/self.num_pred_positive(privileged=privileged)

    def negative_predictive_value(self, privileged=None):
        return self.num_true_negative(privileged=privileged)/self.num_pred_negative(privileged=privileged)

    def false_discovery_rate(self, privileged=None):
        return self.num_false_positive(privileged=privileged)/self.num_pred_positive(privileged=privileged)

    def false_omission_rate(self, privileged=None):
        return self.num_false_negative(privileged=privileged)/self.num_pred_negative(privileged=privileged)

    def sensitivity(self, privileged=None):
        return self.true_positive_rate(privileged=privileged)

    def recall(self, privileged=None):
        return self.true_positive_rate(privileged=privileged)

    def specificity(self, privileged=None):
        return self.true_negative_rate(privileged=privileged)

    def fall_out(self, privileged=None):
        return self.false_positive_rate(privileged=privileged)

    def miss_rate(self, privileged=None):
        return self.false_negative_rate(privileged=privileged)
