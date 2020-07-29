# coding: utf-8

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    print("Import Error: %s"%e)

class AdversarialDebiasing:
    def __init__(self, *args, **kwargs):
        """
        Args:
        - 0: unprivileged_groups (tuple): Representation for unprivileged groups
        - 1: privileged_groups (tuple): Representation for privileged groups
        - 2: scope_name (str): scope name for the tenforflow variables
        - 3: sess (tf.Session): tensorflow session

        Kwargs:
        - seed (int, optional): Seed to make `predict` repeatable.
        - adversary_loss_weight (float, optional): Hyperparameter that chooses
            the strength of the adversarial loss.
        - num_epochs (int, optional): Number of training epochs.
        - batch_size (int, optional): Batch size.
        - classifier_num_hidden_units (int, optional): Number of hidden units
            in the classifier model.
        - debias (bool, optional): Learn a classifier with or without
            debiasing.
        """
        self.unprivileged_groups = args[0]
        self.privileged_groups = args[1]

        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]
        self.scope_name = args[2]
        self.sess = args[3]

        self.seed = kwargs.get('seed', None)
        self.adversary_loss_weight = kwargs.get('adversary_loss_weight', 0.1)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.batch_size = kwargs.get('batch_size', 128)
        self.classifier_num_hidden_units = kwargs.get('classifier_num_hidden_units', 200)
        self.debias = kwargs.get('debias', True)

        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        """
        with tf.variable_scope("classifier_model"):
            W1 = tf.get_variable('W1', [features_dim, self.classifier_num_hidden_units],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name='b1')

            h1 = tf.nn.relu(tf.matmul(features, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

            W2 = tf.get_variable('W2', [self.classifier_num_hidden_units, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_logit = tf.matmul(h1, W2) + b2
            pred_label = tf.sigmoid(pred_logit)

        return pred_label, pred_logit

    def _adversary_model(self, pred_logits, true_labels):
        """Compute the adversary predictions for the protected attribute.
        """
        with tf.variable_scope("adversary_model"):
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

            W2 = tf.get_variable('W2', [3, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_protected_attribute_logit = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

        return pred_protected_attribute_label, pred_protected_attribute_logit
