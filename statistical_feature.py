import scipy.io as sio
import numpy as np


def calculate(filename):
    whole_data = sio.loadmat(filename)
    ave = []
    var = []
    y_count = []

    x = whole_data["x"]
    y = whole_data["y"][0].tolist()
    location = whole_data["location"].tolist()
    l = len(x[0])
    for i in range(l):
        ave.append(round(np.mean(x[:, i]), 2))
        var.append(round(np.var(x[:, i]), 2))

    uniq_ele = list(set(y))
    y_count = [[i, y.count(i) / 100.0] for i in uniq_ele]
    location_y = map(lambda x, y: [x, y], location, y)
    location_y.sort(key=lambda x: x[1])
    label_location = []
    for i in uniq_ele:
        label_location.append(
            map(lambda x: x[0], filter(lambda x: x[1] == i, location_y)))
    # space = 4
    # ind = 0
    # l=len(uniq_ele)
    # while (ind + space < l):
    #     uniq_ele = uniq_ele[:ind + space] + [''] + uniq_ele[ind + space:]
    #     ind = ind + space + 1
    # uniq_ele=str(uniq_ele)

    return [ave, var, y_count, label_location, uniq_ele]
