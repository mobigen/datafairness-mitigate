# coding: utf-8

import numpy as np
import pandas as pd

class ClassificationMetric:
    def __init__(self, origin_df, prediction_df, label_name):
        self.orig_df = origin_df
        self.pred_df = prediction_df
        self.label_name = label_name

        self.origin_pos_mask = (self.orig_df[label_name] == True)
        self.pred_pos_mask = (self.pred_df[label_name] == True)

    def num_positive(self):
        return np.sum(self.origin_pos_mask)

    def num_negative(self):
        return np.sum(~self.origin_pos_mask)

    def num_pred_positive(self):
        return np.sum(self.pred_pos_mask)

    def num_pred_negative(self):
        return np.sum(~self.pred_pos_mask)

    def num_true_positive(self):
        return np.sum(
            (self.orig_df[self.label_name][self.origin_pos_mask] ==
             self.pred_df[self.label_name][self.origin_pos_mask]))

    def num_true_negative(self):
        return np.sum(
            (self.orig_df[self.label_name][~self.origin_pos_mask] ==
             self.pred_df[self.label_name][~self.origin_pos_mask]))

    def num_false_positive(self):
        return np.sum(
            (self.orig_df[self.label_name][~self.origin_pos_mask] !=
             self.pred_df[self.label_name][~self.origin_pos_mask]))

    def num_false_negative(self):
        return np.sum(
            (self.orig_df[self.label_name][self.origin_pos_mask] !=
             self.pred_df[self.label_name][self.origin_pos_mask]))

    def perform_confusion_matrix(self):
        P = self.num_positive()
        N = self.num_negative()
        TP = self.num_true_positive()
        TN = self.num_true_negative()
        FP = self.num_false_positive()
        FN = self.num_false_negative()
        return dict(TPR=TP/P, TNR=TN/N, FPR=FP/N, FNR=FN/P,
                    PPV=TP/(TP+FP), NPV=TN/(TN+FN), FDR=FP/(TP+FP), FOR=FN/(TN+FN),
                    ACC=(TP+TN)/(P+N) if P+N > 0. else np.float64(0.))

    def accuracy(self):
        P = self.num_positive()
        N = self.num_negative()
        TP = self.num_true_positive()
        TN = self.num_true_negative()
        return (TP+TN)/(P+N)

    def true_positive_rate(self):
        return self.num_positive()/self.num_true_positive()

    def true_negative_rate(self):
        return self.num_negative()/self.num_true_negative()

    def false_positive_rate(self):
        return self.num_false_positive()/self.num_negative()

    def false_negative_rate(self):
        return self.num_false_negative()/self.num_positive()

    def positive_predictive_value(self):
        return self.num_true_positive()/self.num_pred_positive()

    def negative_predictive_value(self):
        return self.num_true_negative()/self.num_pred_negative()

    def false_discovery_rate(self):
        return self.num_false_positive()/self.num_pred_positive()

    def false_omission_rate(self):
        return self.num_false_negative()/self.num_pred_negative()

    def sensitivity(self):
        return self.true_positive_rate()

    def recall(self):
        return self.true_positive_rate()

    def specificity(self):
        return self.true_negative_rate()

    def fall_out(self):
        return self.false_positive_rate()

    def miss_rate(self):
        return self.false_negative_rate()
