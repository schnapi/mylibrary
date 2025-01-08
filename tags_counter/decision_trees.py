from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint as sp_randint
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
import graphviz
labels = {'restaurant': 0, 'bar': 1, 'cafe': 2, 'hotel': 3}


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def test_tree(X, y):
    # X = np.array(X)
    # y = np.array(y)
    # data = dict()
    # y1 = []
    # for c in range(4):
    #     data[c] = X[np.where(y == c)][:10]
    #     y1.extend([c] * 10)
    # with open('test1.pickle', 'wb') as f:
    #     pickle.dump([data, y1, X, y], f)
    with open('test1.pickle', 'rb') as f:
        data, y1, X, y = pickle.load(f)
    clf = tree.DecisionTreeClassifier(min_samples_split=3)
    data1 = []
    for v in data.values():
        data1.extend(v)
    data1.append('sandi')
    data1.append('sandi')
    data1.append('dejan sandi')
    data1.append('dejan')
    y1.append(0)
    y1.append(1)
    y1.append(2)
    y1.append(3)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data1, y=list(tags_unique))
    clf = clf.fit(X, np.asarray(y1))
    labels1 = {labels[k]: k for k in labels}
# for c in range(4):
#     for d in data[c]:
#         if 'cuisine_internationa' in d:
#             print(d, labels1[c], labels1[clf.predict(vectorizer.transform([d]))[0]], '\n')
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=vectorizer.get_feature_names(),
                                    class_names=list(labels.keys()),
                                    filled=True, rounded=True,  
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("X1")
    print(clf.tree_.threshold, clf.tree_.impurity)
    pass


with open('cca_prestep.pickle', 'rb') as f:
    lists, tags_unique, y = pickle.load(f)
for l in lists:
    for a in l:
        if 'Meals' in a:
            print(a)
lists[:10]
# with open('balanced_subsample.pickle', 'rb') as f:
#     X1, y = pickle.load(f)
import util as ut
# X1, y = ut.balanced_subsample(lists, list(y), 100)
# # with open('balanced_subsample.pickle', 'rb') as f:
# #     X1, y = pickle.load(f)
# vectorizer = CountVectorizer()
# X1 = [' '.join(x) for x in X1]
# X = vectorizer.fit_transform(X1, y=list(tags_unique))
# test_tree(X1, y)
# exit()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import chi2_kernel, polynomial_kernel, linear_kernel
from sklearn.model_selection import RandomizedSearchCV
cv = StratifiedKFold(n_splits=10)
clf = MultinomialNB(fit_prior=True)
clf = DecisionTreeClassifier()
# find good parameter set
# build a classifier
clf = RandomForestClassifier(n_estimators=500, random_state=0)
# specify parameters and distributions to sample from
param_dist = {
    "max_features": [0.1, 0.3, 0.9],
    "min_samples_split": [53],
    "min_samples_leaf": [3],
    "bootstrap": [True],
    "criterion": ["gini"]}
# param_dist = {  # "max_depth": [3, None],
#     "max_features": np.arange(0.05, 1, 0.1),
#     "min_samples_split": sp_randint(2, 100),
#     "min_samples_leaf": sp_randint(1, 50),
#     "bootstrap": [True],
#     "criterion": ["entropy"]}
# run randomized search
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=cv)
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=3, cv=cv)
random_search.fit(X, y)
report(random_search.cv_results_)
# score = pd.DataFrame(random_search.cv_results_).sort_values(by='mean_validation_score', ascending=False)
# for i in parameters.keys():
#     print(i, len(parameters[i]), parameters[i])
# score[i] = score.parameters.apply(lambda x: x[i])
# l = ['mean_validation_score'] + list(parameters.keys())
# for i in list(parameters.keys()):
#     sns.swarmplot(data=score[l], x=i, y='mean_validation_score')
#     #plt.savefig('170705_sgd_optimisation//'+i+'.jpg', dpi = 100)
#     plt.show()
exit()
# clf.fit(X, y)
# clf = SVC(kernel=polynomial_kernel)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print('accuracy: ' + str(scores.mean()))
exit()
clf1 = Perceptron(fit_intercept=False, tol=None, shuffle=False).fit(X, y)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SGDClassifier(alpha=.0001, penalty="elasticnet")
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['ExtraTreesClassifier', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

pass
