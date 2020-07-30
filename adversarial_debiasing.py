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
        - 0: unprivileged_groups (list of dict):
            - Representation for unprivileged groups.
            - 형식 [{'attr_name': value}, ...] 으로 인자를 받으나 모델 구현때는 리스트의
            index 0에 해당하는 요소를 제외하고 사용하지 않음
        - 1: privileged_groups (list of dict):
            - Representation for privileged groups
            - 형식 [{'attr_name': value}, ...] 으로 인자를 받으나 모델 구현때는 리스트의
            index 0에 해당하는 요소를 제외하고 사용하지 않음
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

    def fit(self, df, protected_attribute_names, label_name):
        """Compute the model parameters of the fair classifier using gradient descent.

        Args:
        - df: pd.DataFrame containing true labels.
        - label_name: label name
        - protected_attribute_names:
            - protected attribute names
            - 실제 학습에 사용되는 protected attribute는 protected attribute names이 여러개라도
            첫번째 name만 사용함

        Returns:
        - AdversarialDebiasing: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        label_ind = df.columns.get_loc(label_name)
        feature_names = np.delete(df.columns.values, label_ind)

        features = df.loc[:,feature_names].values
        protected_attributes = df[protected_attribute_names].values[:, protected_attribute_names.index(self.protected_attribute_name)]
        labels = df[label_name].values

        num_train_samples, self.features_dim = features.shape

        with tf.variable_scope(self.scope_name):
            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph, logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.trainable_variables() if 'classifier_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables() if 'adversary_model' in var.name]
                # Update classifier parameters
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                classifier_grads.append((grad, var))
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars, global_step=global_step)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    batch_features = features[batch_ids]
                    batch_labels = np.reshape(labels[batch_ids], [-1,1])
                    batch_protected_attributes = np.reshape(protected_attributes[batch_ids], [-1,1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                                                                                     pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, pred_labels_loss_value))
        return self
