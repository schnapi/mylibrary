from collections import Counter, defaultdict
from oset import oset
import numpy as np
import scipy.sparse as sp
import random


def balanced_subsample(x, y, subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        indices = [i for i, a in enumerate(y) if a == yi]
        elems = [x[i] for i in indices]
        class_xs.append((yi, elems))
        if min_elems is None or len(elems) < min_elems:
            min_elems = len(elems)

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)
    else:
        use_elems = min(min_elems, int(subsample_size / len(class_xs)))

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = [ci] * use_elems

        xs.extend(x_)
        ys.extend(y_)

    pair = list(zip(xs, ys))
    random.shuffle(pair)
    return [a[0] for a in pair], [a[1] for a in pair]


def _feature_prob_per_tag(clf):
    """Apply smoothing to raw counts and recompute log probabilities"""
    smoothed_fc = clf.feature_count_ + clf.alpha
    smoothed_cc = smoothed_fc.sum(axis=0)
    f_prob_per_tag = smoothed_fc / smoothed_cc
    temp = np.multiply(clf.feature_log_prob_, f_prob_per_tag > 0.75)
    temp[np.where(temp == 0)] = -1000
    return temp


import pickle

with open('naive_bayes_multinominal_vectorize.pickle', 'rb') as f:
    clf, vectorizer = pickle.load(f)
    X = clf.feature_count_


def incidence_matrix(lists, dataset):
    entries, cols, rows = [], [], []
    for i, l in enumerate(lists):
        elem = np.ones(len(l))
        row = np.empty(len(l))
        row.fill(i)
        col = [dataset.index(x) for x in l]
        entries.append(elem)
        rows.append(row)
        cols.append(col)
    return sp.coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))), shape=(len(lists), len(dataset)))


labels = {'restaurant': 0, 'bar': 1, 'cafe': 2, 'hotel': 3}
def incidence_matrix1(lists, dataset, y):
    entries, cols, rows = [], [], []
    additional, addy = [], []
    for i, l in enumerate(lists):
        # if len(l) > 1:
        #     additional.extend(l)
        #     addy.extend([y[i]] * len(l))
        elem = np.ones(len(l))
        row = np.empty(len(l))
        row.fill(i)
        col = [dataset.index(x) for x in l]
        entries.append(elem)
        rows.append(row)
        cols.append(col)
    # maxv = min([len(v) for v in additional.values()])
    # vectorizer = CountVectorizer(lowercase=False)
    # X = vectorizer.fit_transform(additional)

    # for i in range(len(np.where(np.array(addy) == 3)[0])):
    #     entries.append(np.ones(1))
    #     row = np.ones(1) * len(y)
    #     rows.append(row)
    #     cols.append([dataset.index(additional[i])])
    #     y.append(addy[i])
    return sp.coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))), shape=(len(y), len(dataset))), y


def most_informative_feature_for_class(feature_names, classifier, classlabel, n=10, unique=True):
    # prob = classifier.coef_
    prob = _feature_prob_per_tag(classifier)
    features_labels = dict()
    for labelid, label in enumerate(classlabel.keys()):
        topn = sorted(zip(prob[labelid], feature_names))[-n * 40:]
        print(label, list(reversed(topn[-20:])), '\n\n')
        feat = [t[1] for t in topn]
        # for coef, feat in topn:
        #     print(label, feat, coef)
        features_labels[label] = list(reversed(feat))
    return
    res = defaultdict(oset)
    for k, v in features_labels.items():
        to_ = 0
        while len(res[k]) < n:
            to_ += 1
            res[k] = oset(v[:10])
            for k1, v1 in features_labels.items():
                if k == k1:
                    continue
                res[k] -= oset(v1[:to_])
            #     print(v1[0:to_])
            # print(res[k])

        print(k, list(res[k])[:n], to_)
        print('\n')
