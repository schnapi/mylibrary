from collections import Counter, defaultdict
import pickle

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.extmath import density
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from scipy.stats import chi2_contingency
from scipy.stats import chi2 as stat
import scipy.sparse as sp
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
import util as ut

labels = {'restaurant': 0, 'bar': 1, 'cafe': 2, 'hotel': 3}

for a in np.array([1,2,3]):
    print(a)
# with open('BernoulliNB.pickle', 'rb') as f:
#     clf = pickle.load(f)
# with open('naive_bayes_multinominal.pickle', 'rb') as f:
#     vectorizer, X, y = pickle.load(f)

with open('cca_prestep.pickle', 'rb') as f:
    lists, tags_unique, y = pickle.load(f)
# X1, y = ut.balanced_subsample(lists, list(y), 100000)
# with open('balanced_subsample.pickle', 'wb') as f:
#     pickle.dump([X1, y], f)
with open('balanced_subsample.pickle', 'rb') as f:
    X1, y = pickle.load(f)
assert (np.array(y) == 0).sum() == (np.array(y) == 1).sum() == (np.array(y) == 2).sum() == (np.array(y) == 3).sum()
# X, y = ut.incidence_matrix1(X1, list(tags_unique), y)
# vectorizer = TfidfVectorizer(sublinear_tf=False, use_idf=False)
vectorizer = CountVectorizer()
X1 = [' '.join(x) for x in X1]
X = vectorizer.fit_transform(X1, y=list(tags_unique))
# X = ut.incidence_matrix(X1, list(tags_unique))
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=4, init='k-means++', n_init=10, verbose=1)
# clf = BernoulliNB()
# clf = MultinomialNB()
clf = ComplementNB()
cv = KFold(n_splits=3)
scores = cross_val_score(clf, X, y, cv=cv)
clf.fit(X, y)
ut.most_informative_feature_for_class(list(tags_unique), clf, labels, 10)
print('accuracy: ' + str(scores.mean()))
exit()
# scores = cross_validate(clf, X, y, cv=cv, scoring=['accuracy'])

from scipy.sparse import csr_matrix, csc_matrix, coo_matrix


def naive_bayes_multinominal(labels):
    # y = []  # Create target vector
    # text_data = []
    # with open('world.class_from_names_with_tags.txt') as fr:
    #     for l in fr:
    #         temp = l.split(',', 1)
    #         tags = [a.strip() for a in temp[1].split(',')]
    #         text_data += tags
    #         for i in range(len(tags)):
    #             y.append(labels[temp[0].strip()])
    # # Create bag of words
    # vectorizer = CountVectorizer(lowercase=False)
    # X = vectorizer.fit_transform(text_data)  # Create feature matrix
    # with open('naive_bayes_multinominal1.pickle', 'wb') as f:
    #     pickle.dump([vectorizer, X, y, tags_per_class], f)

    with open('naive_bayes_multinominal.pickle', 'rb') as f:
        vectorizer, X, y = pickle.load(f)
    tags_per_class, y1 = [], []
    d = Counter()
    with open('world.class_from_names_with_tags.txt') as fr:
        rowindex = 0
        rows, cols, entries = [], [], []
        classes = {l: {'entries': [], 'rows': [], 'cols': []} for l in labels.keys()}
        for l in fr:
            temp = l.split(',', 1)
            tags = [a.strip() for a in temp[1].split(',')]
            vct = vectorizer.transform(tags).sum(axis=0)[0]
            col = np.where(vct)[1]
            res = d[repr(list(col))]
            d[repr(list(col))] += 1
            if res == 0:
                classes[temp[0]]['entries'].append(np.ones_like(col))
                classes[temp[0]]['rows'].append(np.ones_like(col) * rowindex)
                classes[temp[0]]['cols'].append(col)
                rowindex += 1
        csc = csc_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))), shape=(rowindex, vct.shape[1]))

        pass
    with open('naive_bayes_multinominal1.pickle', 'wb') as f:
        pickle.dump([tags_per_class, y1], f)


def train_chi2(labels):
    with open('data/naive_bayes_multinominal_vectorize.pickle', 'rb') as f:
        clf, vectorizer = pickle.load(f)
        X = clf.feature_count_
        y = labels.values()
    # ch2 = SelectKBest(chi2, k=4)
    # X_train = ch2.fit_transform(X.transpose(), list(range(X.shape[1])))
    c, p, dof, expected = chi2_contingency(X)
    chisq = (X - expected)**2 / expected
    chi2_ = chisq.sum(axis=0)  # chi2 for each column
    pval = stat.sf(chi2_, len(labels) - 1)
    # chi2_, pval = chi2(X, list(y))
    a = sorted(zip(chi2_, pval, range(pval.size)), key=lambda a: a[0], reverse=True)
    best = [i[2] for i in a]
    X1 = (X / X.sum(axis=0) > 0.75).sum(axis=0)  # one class is more than 75%
    X1[np.where((X1 == 0) | (pval != 0))] = 0
    best = (lambda x=X1: [i for i in best if x[i] != 0])()  # remove below 75%
    np.set_printoptions(suppress=True)
    byclass = defaultdict(list)
    temp = X[:, best]
    best = sorted(zip(np.round((temp / temp.sum(axis=0)).max(axis=0) * 100).astype(int),
                      temp.max(axis=0), temp.argmax(axis=0), best), key=lambda x: x[0], reverse=True)
    feature_names = np.array(vectorizer.get_feature_names())
    labels = {labels[k]: k for k in labels}
    byclass = defaultdict(list)
    for maxv, maxc, class_i, i in best:
        byclass[labels[class_i]] += [(maxv, maxc, feature_names[i])]
    for k, v in byclass.items():
        print(k, v, '\n')
    res = defaultdict(oset)

    pass


def train_naive_bayes_multinominal():
    with open('naive_bayes_multinominal.pickle', 'rb') as f:
        vectorizer, X, y = pickle.load(f)
    # Create multinomial naive Bayes object with prior probabilities of each class
    # clf = MultinomialNB(class_prior=[0.25, 0.5])
    clf = MultinomialNB(fit_prior=False)

    # Train model
    clf.fit(X, y)
    # Predict new observation's class
    # model.predict(count.transform(['meals_dinner']))
    with open('naive_bayes_multinominal_vectorize.pickle', 'wb') as f:
        pickle.dump([clf, vectorizer], f)


# train_chi2(labels); exit()
# train_naive_bayes_multinominal()
with open('naive_bayes_multinominal_vectorize.pickle', 'rb') as f:
    clf, vectorizer = pickle.load(f)
clf.class_log_prior_

with open('naive_bayes_multinominal.pickle', 'rb') as f:
    vectorizer, X, y = pickle.load(f)


def calc_MI():
    with open('naive_bayes_multinominal_vectorize.pickle', 'rb') as f:
        clf, vectorizer = pickle.load(f)
        X = clf.feature_count_
        y = labels.values()
    # ch2 = SelectKBest(chi2, k=4)
    # X_train = ch2.fit_transform(X.transpose(), list(range(X.shape[1])))
    feature_names = np.array(vectorizer.get_feature_names())
    g, p, dof, expected = chi2_contingency(X, lambda_="log-likelihood")
    mi = 0.5 * g / X.sum(axis=0)
    misorted = sorted(mi)
    feature_names[mi.argsort()[:10]]
    temp = X[:, mi.argsort()[:10]]
    temp / temp.sum(axis=0)
    pass

    return mi

# calc_MI()


# metrics.adjusted_mutual_info_score(labels_true, labels_pred)
# metrics.homogeneity_score(X, y)
# metrics.completeness_score(X, y)

with open('naive_bayes_multinominal.pickle', 'rb') as f:
    vectorizer, X, y, tags_per_class = pickle.load(f)
with open('world.class_from_names_with_tags.txt') as fr:
    for l in fr:
        temp = l.split(',', 1)
        tags = [a.strip() for a in temp[1].split(',')]
        labels[temp[0].strip()]
        temp = vectorizer.transform(tags)
        mydict = defaultdict(lambda: 0)
        print(clf.predict_proba(temp))
        tags_per_class
        for argmax, val in zip(clf.predict_proba(temp).argmax(axis=1), clf.predict_proba(temp).max(axis=1)):
            mydict[argmax] += val
        max(mydict.keys(), key=(lambda k: mydict[k]))
        clf.feature_count_
        pass


# most_informative_feature_for_class(vectorizer, clf, labels, 20)

pass
