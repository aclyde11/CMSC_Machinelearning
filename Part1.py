from __future__ import print_function

import sys

import pandas as pd
import numpy as np
import tabulate
import argparse
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def arg_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    return parser.parse_args()

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


def part1_run_model_and_report_class_stats(X, Y, clf, CV=5):
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
        clf = RandomForestClassifier(n_estimators=250, criterion='entropy', n_jobs=-1)
        scores = part1_run_model_and_report_class_stats(X, Y, clf)
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
        score = part1_run_model_and_report_class_stats(X_coding_scaled, Y, clf)
        scores.append(score.sort_values("acc").iloc[0])
    print(tabulate.tabulate(pd.DataFrame(scores, index=names).sort_values("acc"), headers='keys', tablefmt='psql'))


def part_c(X_all, X_coding, Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFECV
    from sklearn.model_selection import StratifiedKFold

    mi_sel =  SelectKBest(mutual_info_classif, k=100)
    X_reduced = mi_sel.fit_transform(X_coding, Y)

    clf = RandomForestClassifier(n_estimators=250, criterion='entropy', n_jobs=8)
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2, shuffle=True), scoring='accuracy', n_jobs=4)
    rfecv.fit(X_reduced, Y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig('part1_c_feature_sel.png')
    plt.show()


def part_d_run_model(X_coding, Y, Y_holdout, X_holdout):
    from sklearn import metrics
    import keras
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model

    input = Input((X_coding.shape[1],))
    x = Dense(100, activation='relu', kernel_initializer='uniform')(input)
    x = Dense(50, activation='relu', kernel_initializer='uniform')(x)
    x = Dense(50, activation='relu', kernel_initializer='uniform')(x)
    x = Dense(2, activation='softmax', kernel_initializer='uniform')(x)

    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_train = keras.utils.to_categorical(Y, 2)
    model.fit(X_coding, y_train, validation_split=0, epochs=50, batch_size=32, verbose=0)

    Y_predict = model.predict(X_holdout)
    Y_holdout = keras.utils.to_categorical(Y_holdout, 2)

    threshold = 0.5
    Y_pred_int = (Y_predict[:, 0] < threshold).astype(np.int)
    Y_test_int = (Y_holdout[:, 0] < threshold).astype(np.int)

    score_dict = {}
    score_dict['acc'] = metrics.accuracy_score(Y_test_int, Y_pred_int)
    score_dict['r2'] = metrics.r2_score(Y_test_int, Y_pred_int)
    score_dict['confusion'] = metrics.confusion_matrix(Y_test_int, Y_pred_int)
    score_dict['f1'] = metrics.f1_score(Y_test_int, Y_pred_int)
    return score_dict


def part_d(X_all, X_coding, Y):
    # TODO: Implement CV for accuacy score.... otherwise plot is bad
    from sklearn import preprocessing
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split

    print("Scaling coding data.")
    X_coding_scaled = preprocessing.StandardScaler().fit_transform(X_coding)
    X_train, X_test, y_train, y_test = train_test_split(X_coding_scaled, Y, stratify=Y, train_size=0.65)

    print("Testing 10 different training set sizes.")
    iters = 15
    scores = []
    sizes = np.linspace(35, X_train.shape[0] - 10, num=iters, dtype=int)
    for size in sizes:
        print("Running size %i" % size)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=size-2, test_size=2,random_state=30)
        train_index, _ = list(sss.split(X_train, y_train))[0]
        X_sub = X_train[train_index,:]
        Y_sub = y_train[train_index]
        scores.append(part_d_run_model(X_sub, Y_sub, y_test, X_test))

    df = pd.DataFrame(scores)
    print(tabulate.tabulate(df, headers='keys', tablefmt='psql'))
    plt.plot(sizes, df['acc'])
    plt.xlabel("Size of training set")
    plt.ylabel("Accuracy on holdout")
    plt.title("Learning Curve for NT Problem")
    plt.savefig('part1_d_learning_curve.png')
    plt.show()

def part_1_ec(X_all, X_coding, Y):
    from tpot import TPOTClassifier
    from sklearn.model_selection import train_test_split
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,n_jobs=16,
                                        random_state=42, verbosity=2, max_time_mins=180)

    X_train, X_test, y_train, y_test = train_test_split(X_coding, Y, stratify=Y, random_state=42)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))


if __name__ == "__main__":
    args = arg_setup()
    X_coding, Y_coding = load_data(args.data + "nt.coding.csv")
    X_all, Y_all = load_data(args.data + "nt.all.csv")

    Y = None
    if np.all(Y_all != Y_coding):
        print("Assertion that I had in mind failed. Exiting")
        exit(0)
    else:
        Y = Y_coding
        del Y_all
        del Y_coding

    print("PART 1: Normal versus Tumor Predictions")
    for part in 'abcd':
        print_heading("PART %s" % part)
        locals()["part_%s" % part](X_all, X_coding, Y)
    part_1_ec(X_all, X_coding, Y)
