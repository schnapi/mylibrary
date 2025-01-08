import numpy as np
import scipy.sparse as sp
import pickle
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def incidence_matrix(lists, dataset):
    entries = []
    cols = []
    rows = []
    for i, l in enumerate(lists):
        elem = np.ones(len(l))
        row = np.empty(len(l))
        row.fill(i)
        col = [dataset.index(x) for x in l]
        entries.append(elem)
        rows.append(row)
        cols.append(col)
    e = np.concatenate(entries)
    c = np.concatenate(cols)
    r = np.concatenate(rows)
    len(r)
    return sp.coo_matrix((e, (r, c)), shape=(len(lists), len(dataset)))


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    Y = [[i] for i, y in enumerate(Y)]
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=4).fit(X.toarray(), Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


# labels = {'restaurant': 0, 'bar': 1, 'cafe': 2, 'hotel': 3}
# y = []
# lists = []
# tags_unique = set()
# with open('world.class_from_names_with_tags.txt') as fr:
#     index = 0
#     for l in fr:
#         index += 1
#         if index > 900000:
#             break
#         temp = l.split(',', 1)
#         tags = [a.strip() for a in temp[1].split(',')]
#         lists += [tags]
#         tags_unique.update(tags)
#         y.append(labels[temp[0].strip()])
#     with open('cca_prestep.pickle', 'wb') as f:
#         pickle.dump([lists, tags_unique, y], f)
# exit()
with open('cca_prestep.pickle', 'rb') as f:
    lists, tags_unique, y = pickle.load(f)
X = incidence_matrix(lists[:100], list(tags_unique))
plt.figure(figsize=(8, 6))

plot_subfigure(X, y, 1, "With unlabeled samples + CCA", "cca")
# plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()
