from __future__ import print_function

import sys

import pandas as pd
import numpy as np
import tabulate

from tqdm import tqdm
import matplotlib.pyplot as plt


def print_heading(msg):
    width = len(msg) + 4
    for i in range(width):
        print("#", end='')
    print()
    print("# %s #" % msg)
    for i in range(width):
        print("#", end='')
    print()


def load_data(file):
    df = pd.read_csv(file, low_memory=False, engine='c')
    X = df.values[:, 1:].astype(np.float32)
    y = df.values[:, 0].astype(np.int32)
    return X, y


def run_model_and_report_class_stats(X, Y, clf, CV=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics

    cv = StratifiedKFold(CV, shuffle=True)
    scores = pd.DataFrame(columns=['acc', 'r2', 'confusion', 'f1'])
    for train_index, test_index in cv.split(X, Y):
        clf.fit(X[train_index, :], Y[train_index])

        y_pred = clf.predict(X[test_index, :])

        score_dict = {}
        score_dict['acc'] = clf.score(X[test_index, :], Y[test_index])
        score_dict['r2'] = metrics.r2_score(Y[test_index], y_pred)
        score_dict['confusion'] = metrics.confusion_matrix(Y[test_index], y_pred)
        score_dict['f1'] = metrics.f1_score(Y[test_index], y_pred)

        scores = pd.concat([scores, pd.DataFrame([score_dict], columns=score_dict.keys())], sort=True)
    return scores.reset_index().drop("index", axis=1)


def part_a(X_all, X_coding, Y):
    from sklearn.ensemble import RandomForestClassifier

    print("Using RandomForestClassifer with 500 estimators and 5-fold StratifiedKFold.")
    print("No normalization or scaling is applied (RF doesn't need it).")
    for X, type in [(X_all, 'all'), (X_coding, 'coding')]:
        print("Using %s as input features." % type)
        clf = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
        scores = run_model_and_report_class_stats(X, Y, clf)
        print(tabulate.tabulate(scores, headers='keys', tablefmt='psql'))


def part_b(X_all, X_coding, Y):
    from sklearn import preprocessing
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # This list of classifers is stolen directly from the example on Sklearn "Classifcation comparison"
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(4, n_jobs=-1),
        SVC(kernel="linear"),
        SVC(kernel="rbf", gamma='auto'),
        GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=-1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    scores = []

    print("Scaling coding data.")
    X_coding_scaled = preprocessing.StandardScaler().fit_transform(X_coding)
    print("Performing search through 10 different classifcations algos using X_coding...")
    for name, clf in tqdm(zip(names, classifiers)):
        score = run_model_and_report_class_stats(X_coding_scaled, Y, clf)
        scores.append(score.sort_values("acc").iloc[0])
    print(tabulate.tabulate(pd.DataFrame(scores, index=names).sort_values("acc"), headers='keys', tablefmt='psql'))


def part_c(X_all, X_coding, Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFECV
    from sklearn.model_selection import StratifiedKFold
    X_reduced = SelectKBest(mutual_info_classif, k=100).fit_transform(X_coding, Y)

    clf = RandomForestClassifier(n_estimators=250, criterion='entropy', n_jobs=-1)
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=2, cv=StratifiedKFold(5, shuffle=True), scoring='accuracy')
    rfecv.fit(X_reduced, Y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    # TODO: print gene signature


def part_d_run_model(X_all, X_coding, Y):
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import keras
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model

    input = Input((X_coding.shape[1],))
    x = Dense(100, activation='tanh', kernel_initializer='uniform')(input)
    x = Dense(100, activation='tanh', kernel_initializer='uniform')(x)
    x = Dense(100, activation='tanh', kernel_initializer='uniform')(x)
    x = Dense(2, activation='softmax', kernel_initializer='unifiorm')(x)

    lr_sched = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto',
                                                 min_delta=0.001,
                                                 cooldown=0, min_lr=0)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    X_train, X_train, X_test, y_test = train_test_split(X_coding, Y, stratify=Y)
    model.fit(X_train, X_train, validation_data=[X_test, y_test], epochs=100, callbacks=[lr_sched], batch_size=32)

    Y_predict = model.predict(X_test)
    threshold = 0.5
    Y_pred_int = (Y_predict[:, 0] < threshold).astype(np.int)
    Y_test_int = (y_test[:, 0] < threshold).astype(np.int)

    score_dict = {}
    score_dict['acc'] = metrics.accuracy_score(Y_test_int, Y_pred_int)
    score_dict['r2'] = metrics.r2_score(Y_test_int, Y_pred_int)
    score_dict['confusion'] = metrics.confusion_matrix(Y_test_int, Y_pred_int)
    score_dict['f1'] = metrics.f1_score(Y_test_int, Y_pred_int)
    return score_dict


def part_d(X_all, X_coding, Y):
    learning_set_sizes = []
    scores = []
    for size in learning_set_sizes:
        scores.append(part_d_run_model(X_all, X_coding, Y))

    # TODO: plot learning curve


if __name__ == "__main__":
    X_coding, Y_coding = load_data("/Volumes/ExternalClyde/machine_learning_for_cancer_work/hw1/nt.coding.csv")
    X_all, Y_all = load_data("/Volumes/ExternalClyde/machine_learning_for_cancer_work/hw1/nt.all.csv")

    Y = None
    if np.all(Y_all != Y_coding):
        print("Assertion that I had in mind failed. Exiting")
        exit(0)
    else:
        Y = Y_coding
        del Y_all
        del Y_coding

    for part in 'abcd':
        print_heading("PART %s" % part)
        locals()["part_%s" % part](X_all, X_coding, Y)
