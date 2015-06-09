import scipy.io as sio
import numpy as np
from statsmodels.stats.inter_rater import cohens_kappa
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import random
import os

static_path = "static/"


def predict(filename, clf, selected_feature):
    # read data
    all_data = sio.loadmat(filename)
    x_tr = all_data["x_tr"].tolist()
    y_tr = all_data["y_tr"][0].tolist()
    x_te = all_data["x_te"].tolist()
    y_te = all_data["y_te"][0].tolist()
    te_location = all_data["te_location"].tolist()

    if selected_feature != "":
        sf = selected_feature.split(",")
        sf = map(lambda x: int(x), sf)
        for i in range(len(x_tr)):
            new_feature=[]
            for j in sf:
                new_feature.append(x_tr[i][j])
            x_tr[i]=new_feature

        for i in range(len(x_te)):
            new_feature=[]
            for j in sf:
                new_feature.append(x_te[i][j])
            x_te[i]=new_feature

    # fit and predict
    clf.fit(np.array(x_tr),np.array(y_tr))
    print "here"
    testing_result = clf.predict(x_te).tolist()
    print "predict done"

    # overall accuracy
    OA = float(sum([1 for i in range(len(y_te))
                    if testing_result[i] == y_te[i]])) / len(y_te)

    # count of each label,for average accuracy
    label_count = {}
    for label in y_te:
        if label in label_count:
            label_count[label] = label_count[label] + 1
        else:
            label_count[label] = 1
    # correct classification in each label
    label_correct_count = {}
    # kappa matrix for calculate kappa statistics
    kappa_matrix = {}
    for label in label_count:
        label_correct_count[label] = 0
        kappa_matrix[label] = {}
        for alabel in label_count:
            kappa_matrix[label][alabel] = 0

    for i in range(len(y_te)):
        if y_te[i] == int(testing_result[i]):
            # record correct classification
            label_correct_count[y_te[i]] = label_correct_count[y_te[i]] + 1
        kappa_matrix[y_te[i]][int(testing_result[i])] = kappa_matrix[
            y_te[i]][int(testing_result[i])] + 1

    # accuracy of each label
    label_accuracy = {}
    for label in label_count:
        label_accuracy[label] = float(label_correct_count[
            label]) / label_count[label]

    # average accuracy
    AA = 0
    for label in label_accuracy:
        AA = AA + label_accuracy[label]
    AA = AA / len(label_accuracy)

    # kappa statistics
    kappa_matrix_list = [[0 for j in label_count] for i in label_count]
    i = 0
    for label in kappa_matrix:
        j = 0
        for alabel in kappa_matrix[label]:
            kappa_matrix_list[i][j] = kappa_matrix[label][alabel]
            j = j + 1
        i = i + 1

    kappa = cohens_kappa(np.array(kappa_matrix_list)).kappa

    # label
    uniq_ele = list(set(y_te))

    # get true data' location and label
    y_location = map(lambda x, y: [x, y], te_location, y_te)
    # sorted by label
    y_location.sort(key=lambda x: x[1])
    # get location of different label in independent list
    y_te_location = []
    for i in uniq_ele:
        y_te_location.append(
            map(lambda x: x[0], filter(lambda x: x[1] == i, y_location)))

    # get testing result' location and label
    y_location = map(lambda x, y: [x, y], te_location, testing_result)
    # print y_location
    y_location.sort(key=lambda x: x[1])
    t_te_location = []
    for i in uniq_ele:
        t_te_location.append(
            map(lambda x: x[0], filter(lambda x: x[1] == i, y_location)))

    # for i in tr_location:
    #     if i ==[]:
    #         print i

    # f = open("tempresult", "w")
    # f.write(str(label_accuracy))
    # f.close()

    filename = "result/result.mat"
    while os.path.exists(filename):
        filename = filename[:-4] + str(random.randint(1, 9)) + filename[-4:]
    # print filename
    sio.savemat(
        filename, {"testing_result": testing_result, "true_result": y_te, "location": te_location})

    return [OA * 100, AA * 100, kappa * 100, label_accuracy, y_te_location, t_te_location, uniq_ele, filename]


def calculate(filename,selected_feature, method="SVM", kernel="poly", C=1.0, gamma=0.0):
    if method == "Logistic":
        clf = LogisticRegression()
    elif method == "SVM":
        if kernel == "RBF":
            clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)
        elif kernel == "Linear":
            clf = svm.SVC(kernel="linear")
        else:
            clf = svm.SVC(kernel="poly")
    return predict(filename, clf, selected_feature)
