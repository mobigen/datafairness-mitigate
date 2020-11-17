# coding: utf-8

# Import Packages
import tensorflow as tf

from utils import datasets, preproc
from metric import DatasetMetric, ClassificationMetric
from algorithm.adversarial_debiasing import AdversarialDebiasing

def main():
    print('Tensorflow Version:', tf.__version__)

    # 1. Dataset
    # 1-1. Load & Preprocessing
    #   Set Columns and Protected Atrributes
    #   18 Features including 2 Protected Attributes and 1 Label
    protected_attribute_names = ['sex', 'race']
    privileged_classes = [['Male'], ['White']]
    label_name = 'income-per-year'
    one_hot_features = ['Age (decade)', 'Education Years']

    #   Load Test Dataset & Preprocess
    df_orig = datasets.get_adults_df()
    df_orig = preproc.preprocess_df(df_orig,
                                     protected_attribute_names, privileged_classes,
                                     label_name, ['>50K', '>50K.'],
                                     one_hot_column_names=one_hot_features)

    print('#### Basic Statistics of DataFrame')
    preproc.describe_df(df_orig, detail=True)

    # 1-2.Train Test Splitting
    #   Split Dataset to train:test=7:3
    df_orig_train, df_orig_test = preproc.split_df(df_orig)

    print('#### Trainset')
    preproc.describe_df(df_orig_train)
    print('#### Testset')
    preproc.describe_df(df_orig_test)

    # 1-3. Fairness Metric of Origin Data
    #   데이터셋 자체에 대한 Metric
    dataset_metric_train_without_debias = DatasetMetric(df_orig_train, 'sex', label_name)
    dataset_metric_test_without_debias = DatasetMetric(df_orig_test, 'sex', label_name)

    print('#### Original Dataset Metric')

    print('#### - Base Rate')
    print('Unprivileged, Privileged Group 에서 각각 Positive가 차지하는 비율')
    print('Train Set, Unprivileged Group: %f' % dataset_metric_train_without_debias.base_rate(privileged=False))
    print(' Test Set, Unprivileged Group: %f' % dataset_metric_test_without_debias.base_rate(privileged=False))
    print('Train Set, Privileged Group: %f' % dataset_metric_train_without_debias.base_rate(privileged=True))
    print(' Test Set, Privileged Group: %f' % dataset_metric_test_without_debias.base_rate(privileged=True))

    print('#### - Mean Difference')
    print('Unprivileged, Privileged Group 간 Base Rate의 차')
    print('Train Set: Mean Difference between Unprivileged and Privileged Group = %f' %
          dataset_metric_train_without_debias.mean_difference())
    print(' Test Set: Mean Difference between Unprivileged and Privileged Group = %f' %
          dataset_metric_test_without_debias.mean_difference())

    # 2. Learn Classifier with Adversarial Debiasing
    # 2-1. Model without Debiasing
    #   Debiasing을 하지 않는 Plain Model
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    sess = tf.Session()
    plain_model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                       'plain_classifier', sess, debias=False)
    plain_model.fit(df_orig_train, protected_attribute_names, label_name)

    #   학습된 Plain Model을 Test Data에 적용
    df_pred_train_nodebiased = plain_model.predict(df_orig_train, label_name)
    df_pred_test_nodebiased = plain_model.predict(df_orig_test, label_name)

    # 2-2. Model with Debiasing
    #   Debiasing을 수행하는 Debiased Model
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                          'debiased_classifier', sess, debias=True)
    debiased_model.fit(df_orig_train, protected_attribute_names, label_name)

    #   학습된 Debiased Model을 Test Data에 적용
    df_pred_train_debiased = debiased_model.predict(df_orig_train, label_name)
    df_pred_test_debiased = debiased_model.predict(df_orig_test, label_name)

    # 3. Fairness Metrics
    #   Debiasing 전후 Dataset과 Model 성능 비교를 위한 Metric 측정
    dataset_metric_train_without_debias = DatasetMetric(df_orig_train, 'sex', label_name)
    dataset_metric_test_without_debias = DatasetMetric(df_orig_test, 'sex', label_name)

    classified_metric_train_without_debias = ClassificationMetric(df_orig_train,
                                                                  df_pred_train_nodebiased,
                                                                  'sex', label_name)
    classified_metric_test_without_debias = ClassificationMetric(df_orig_test,
                                                                 df_pred_test_nodebiased,
                                                                 'sex', label_name)

    dataset_metric_train_with_debias = DatasetMetric(df_pred_train_debiased, 'sex', label_name)
    dataset_metric_test_with_debias = DatasetMetric(df_pred_test_debiased, 'sex', label_name)

    classified_metric_train_with_debias = ClassificationMetric(df_orig_train,
                                                               df_pred_train_debiased,
                                                               'sex', label_name)
    classified_metric_test_with_debias = ClassificationMetric(df_orig_test,
                                                              df_pred_test_debiased,
                                                              'sex', label_name)

    def explain_metric(met):
        print('Accuracy: ', met.accuracy())
        print('Balanced Accuray: ', met.balanced_accuracy())
        print('Disparate Impact: ', met.disparate_impact())
        print('Equal Opportunity Difference: ', met.equal_opportunity_difference())
        print('Average Odds Difference: ', met.average_odds_difference())
        print('Theil Index: ', met.theil_index())

    print('#### Dataset Metric - Original Dataset')
    print('Train Set: Mean Difference between Unprivileged and Privileged Group = %f' %
          dataset_metric_train_without_debias.mean_difference())
    print(' Test Set: Mean Difference between Unprivileged and Privileged Group = %f' %
          dataset_metric_test_without_debias.mean_difference())

    print('#### Dataset Metric - Debiased Dataset')
    print('Train Set: Mean Difference between Unprivileged and Privileged Group = %f' %
          dataset_metric_train_with_debias.mean_difference())
    print(' Test Set: Mean Difference between Unprivileged and Privileged Group = %f' %
          dataset_metric_test_with_debias.mean_difference())

    print('#### Classification Metric - Plain Model - Test Dataset')
    explain_metric(classified_metric_test_without_debias)

    print('#### Classification Metric - Debiased Model - Test Dataset')
    explain_metric(classified_metric_test_with_debias)

    # References:
    #
    # 1. B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating UnwantedBiases with Adversarial Learning,"
    #    AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.
    #
    # 2. https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_adversarial_debiasing.ipynb

if __name__ == '__main__':
    main()
