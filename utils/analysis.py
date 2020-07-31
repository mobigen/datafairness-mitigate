# coding: utf-8

import numpy as np
import pandas as pd

class ClassificationMetric:
    def __init__(self, origin_df, prediction_df, label_name):
        self.orig_df = origin_df
        self.pred_df = prediction_df
        self.label_name = label_name

        self.origin_pos_mask = (self.orig_df[label_name] == True)

    def num_positive(self):
        return np.sum(self.origin_pos_mask)

    def num_negative(self):
        return np.sum(~self.origin_pos_mask)

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
                    ACC=(TP+TN)/(P+N) if P+N > 0. else np.float64(0.))

    def accuracy(self):
        P = self.num_positive()
        N = self.num_negative()
        TP = self.num_true_positive()
        TN = self.num_true_negative()
        return (TP+TN)/(P+N)
