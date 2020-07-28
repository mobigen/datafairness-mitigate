# coding: utf-8

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    print("Import Error: %s"%e)

class AdversarialDebiasing:
    def __init__(self,
                 unprivileged_groups,
                 privileged_groups,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=50,
                 batch_size=128,
                 classifier_num_hidden_units=200,
                 debias=True):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged groups
            privileged_groups (tuple): Representation for privileged groups
            seed (int, optional): Seed to make `predict` repeatable.
            adversary_loss_weight (float, optional): Hyperparameter that chooses
                the strength of the adversarial loss.
            num_epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier model.
            debias (bool, optional): Learn a classifier with or without
                debiasing.
        """
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias

        self.features_dim = None
        self.protected_attributes = None
        self.true_labels = None
        self.pred_labels = None

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        """
        pass
