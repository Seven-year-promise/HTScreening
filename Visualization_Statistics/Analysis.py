import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from utils import CLASSES, load_effected_data_feature_median, exist_key, get_key
from scipy.stats import f_oneway, ttest_ind
from sklearn.feature_selection import f_classif
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from statsmodels.multivariate.manova import MANOVA

def t_test_selected(data, actions, selected_a):
    wild_data = data[actions == 0, :]
    action_data = data[actions == selected_a, :]
    f, p = ttest_ind(wild_data, action_data, equal_var=False)
    return f, p

def f_one_way_test_selected(data, actions, selected_a):
    wild_data = data[actions==0, :]
    action_data = data[actions == selected_a, :]

    F, p = f_oneway(wild_data, action_data)

    return F, p

def f_one_way_test(data, actions):
    action0_data = data[actions==0, :]
    action1_data = data[actions == 1, :]
    action2_data = data[actions == 2, :]
    action3_data = data[actions == 3, :]

    F, p = f_oneway(action0_data, action1_data, action2_data, action3_data)

    return F, p

def MANOVA_test_selected(data, actions, selected_a):
    inds = actions==0
    inds = np.logical_or(actions==selected_a, inds)

    actions = actions.reshape(-1, 1)
    res = MANOVA(endog=data[inds], exog=actions[inds])
    print(res.mv_test().summary())
    return res

def MANOVA_test(data, actions):
    actions = actions.reshape(-1, 1)
    res = MANOVA(endog=data, exog=actions)
    print(res.mv_test().summary())
    return res

def outlier_remove(data, actions):
    action_num = np.max(actions)
    #print(action_num)
    filtered_data = []
    filtered_actions= []
    for a in range(action_num+1):
        action_data = data[actions == a, :]
        #print(data.shape, action_data.shape)
        clf = LocalOutlierFactor(n_neighbors=500, p=2)
        #clf = EllipticEnvelope(random_state=0).fit(action_data)

        pred = clf.fit_predict(action_data)
        #pred = clf.predict(action_data)
        #print(pred)
        filtered_action_data = action_data[pred==1, :]
        #print(filtered_action_data.shape)
        for n in range(filtered_action_data.shape[0]):
            filtered_data.append(filtered_action_data[n, :])
            filtered_actions.append(a)
    return np.array(filtered_data), np.array(filtered_actions)

def feature_normalize(data):
    return normalize(data, axis=0)

if __name__ == "__main__":
    data, comps, actions = load_effected_data_feature_median(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    f_one_way_test(data, actions)
