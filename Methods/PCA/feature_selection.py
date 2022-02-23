import numpy as np


def median(feature={}, labels={}, num_class=130):
    median_features = {}
    median_features_train = []
    median_features_eval = []
    feature_labels = {}
    feature_labels_train = []
    feature_labels_eval = []
    for c in range(num_class):
        # ind_vec = np.zeros_like(classes)
        # ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = feature["train"][train_inds]
        eval_comp_data = feature["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue

        median_features_train.append(np.median(train_comp_data, axis=0))
        median_features_eval.append(np.median(eval_comp_data, axis=0))
        feature_labels_train.append(c)
        feature_labels_eval.append(c)

    median_features["train"] = np.array(median_features_train)
    median_features["eval"] = np.array(median_features_eval)
    feature_labels["train"] = np.array(feature_labels_train)
    feature_labels["eval"] = np.array(feature_labels_eval)

    return median_features, feature_labels


def mean(feature={}, labels={}, num_class=130):
    mean_features = {}
    mean_features_train = []
    mean_features_eval = []
    feature_labels = {}
    feature_labels_train = []
    feature_labels_eval = []
    for c in range(num_class):
        # ind_vec = np.zeros_like(classes)
        # ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = feature["train"][train_inds]
        eval_comp_data = feature["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue

        mean_features_train.append(np.average(train_comp_data, axis=0))
        mean_features_eval.append(np.average(eval_comp_data, axis=0))
        feature_labels_train.append(c)
        feature_labels_eval.append(c)

    mean_features["train"] = np.array(mean_features_train)
    mean_features["eval"] = np.array(mean_features_eval)
    feature_labels["train"] = np.array(feature_labels_train)
    feature_labels["eval"] = np.array(feature_labels_eval)

    return mean_features, feature_labels