import scipy.io as sio
import numpy as np
import threading
from sklearn import cross_validation
from sklearn import svm
import time


class TaskTable(object):
    # task dictionary ,session:task
    tasks = {}

    def register(self, session, task):
        # register task with uniq session
        # print session
        self.tasks[session] = task
        # print self.tasks

    def unregister(self, session):
        # unregister task with uniq session
        if session not in self.tasks:
            return
        self.tasks[session].running = 0
        del(self.tasks[session])

    def getprogress(self, session):
        # print session
        # print self.tasks
        # print session not in self.tasks
        if session not in self.tasks:
            return -1
        # delete from tasks
        return self.tasks[session].getprogress()

    def getresult(self, session):
        # get progress with uniq session
        # print session
        # print self.tasks
        # print session not in self.tasks
        if session not in self.tasks:
            return -1
        return self.tasks[session].getresult()


class RBFGammaTask(threading.Thread):

    def __init__(self, filename, min_g, max_g, step, C):
        threading.Thread.__init__(self)
        data = sio.loadmat(filename)
        self.x = data["x"]
        self.y = data["y"][0]
        self.min_g = min_g
        self.max_g = max_g
        self.C = C
        self.step = step
        # acc of different gamma
        self.result = []
        self.running = 1
        # record the count of gamma that have done
        self.done = 0
        # total count of calculation
        self.total = len(np.arange(min_g, max_g, step))

    def run(self):
        for gamma in np.arange(self.min_g, self.max_g, self.step):
            if self.running == 0:
                break
            if gamma == 0.0:
                gamma = "auto"
            clf = svm.SVC(kernel="rbf", gamma=gamma, C=self.C)
            score = round(cross_validation.cross_val_score(
                clf, self.x, self.y, cv=5).mean() * 100, 2)
            # record result
            self.result.append(score)
            self.done = self.done + 1

    def getprogress(self):
        # how many have we done
        return float(self.done) / self.total

    def getresult(self):
        # not done yet
        if self.done != self.total:
            return [-1, -1]
        else:
            # in this task ,return strings of different gamma and accuracy of
            # different gamma
            return [str(np.arange(self.min_g, self.max_g, self.step).tolist())[1:-1], str(self.result)[1:-1]]


class SoftMarginTask(threading.Thread):

    def __init__(self, filename, min_sf, max_sf, step, kernel, gamma):
        threading.Thread.__init__(self)
        data = sio.loadmat(filename)
        self.x = data["x"]
        self.y = data["y"][0]
        self.min_sf = min_sf
        self.max_sf = max_sf
        self.step = step

        if kernel == "RBF":
            self.clf = svm.SVC(kernel="rbf", gamma=gamma)
        elif kernel == "Linear":
            self.clf = svm.SVC(kernel="linear")
        else:
            self.clf = svm.SVC(kernel="poly")

        # acc of different gamma
        self.result = []
        self.running = 1
        # record the count of gamma that have done
        self.done = 0
        # total count of calculation
        self.total = len(np.arange(min_sf, max_sf, step))

        # print "************"
        # print self.done
        # print self.total

    def run(self):
        for C in np.arange(self.min_sf, self.max_sf, self.step):
            if self.running == 0:
                break
            self.clf.set_params(C=C)
            # print "***************"
            # print C
            # print self.clf
            score = round(cross_validation.cross_val_score(
                self.clf, self.x, self.y, cv=5).mean() * 100, 2)
            # record result
            self.result.append(score)
            self.done = self.done + 1

    def getprogress(self):
        # how many have we done
        return float(self.done) / self.total

    def getresult(self):
        # not done yet
        if self.done != self.total:
            return [-1, -1]
        else:
            # in this task ,return strings of different gamma and accuracy of
            # different gamma
            return [str(np.arange(self.min_sf, self.max_sf, self.step).tolist())[1:-1], str(self.result)[1:-1]]


class ForwardStepwiseTask(threading.Thread):

    def __init__(self, filename, n_features):
        threading.Thread.__init__(self)
        data = sio.loadmat(filename)
        self.x = data["x"]
        self.y = data["y"][0]
        self.bandlength = len(self.x[0])
        self.n_features = n_features

        if self.bandlength < self.n_features:
            raise Exception("n_features is too large")

        self.clf = svm.SVC(kernel="rbf")
        # acc of different gamma
        self.result = []
        self.acc = []
        self.running = 1
        # record the count of gamma that have done
        self.done = 0
        # total count of calculation
        self.total = n_features

        # print "************"
        # print self.done
        # print self.total

    def run(self):
        for i in range(self.n_features):
            acc = []
            for feature in range(self.bandlength):
                if feature not in self.result:
                    acc.append([feature, self.cal_cv(feature)])
            acc.sort(key=lambda x: -x[1])
            self.result.append(acc[0][0])
            self.acc.append(acc[0][1])
            self.done = self.done + 1

    def cal_cv(self, feature):
        xx = []
        for i in range(len(self.x)):
            features = [self.x[i][j] for j in self.result]
            features.append(self.x[i][feature])
            xx.append(features)
        return round(cross_validation.cross_val_score(
            self.clf, xx, self.y, cv=5).mean() * 100, 2)

    def getprogress(self):
        # how many have we done
        return float(self.done) / self.total

    def getresult(self):
        # not done yet
        if self.done != self.total:
            return [-1, -1]
        else:
            # in this task ,return strings of different gamma and accuracy of
            # different gamma
            return [str(self.result)[1:-1], str(self.acc)[1:-1]]


class KnnTask(threading.Thread):

    def __init__(self, filename, margin):
        threading.Thread.__init__(self)
        data = sio.loadmat(filename)
        self.y = data["testing_result"][0]
        self.location = data["location"].tolist()
        self.true_result=data["true_result"][0]
        self.margin = margin

        # acc of different gamma
        self.result = []
        self.traks = []
        self.running = 1
        # record the count of gamma that have done
        self.done = 0
        # total count of calculation
        self.total = len(self.y)+1

    def run(self):
        st = time.time()
        self.cal_new_result() 
        max_times = (2 * self.margin + 1) ** 2
        for i in range(max_times):
            [acc, trak] = self.cal_with_least_times(i)
            self.result.append(acc)
            self.traks.append(trak)
            # print self.result
        self.done = self.done + 1

    def cal_new_result(self):
        rounds = [[i, j] for i in range(-self.margin, self.margin + 1, 1)
                  for j in range(-self.margin, self.margin + 1, 1) if not (i == 0 and j == 0)]
        # print rounds

        self.max_label_count = []
        # random_set = random.sample(range(len(testing_result)), len(testing_result))
        for i in range(len(self.y)):
            [x, y] = self.location[i]
            new_result = []
            count = 0
            for j in rounds:
                new_x = x + j[0]
                new_y = y + j[1]

                # OA:78.69%
                if [new_x, new_y] in self.location:
                    ind = self.location.index([new_x, new_y])
                    new_result.append(self.y[ind])
                    count = count + 1

            if count != 0:
                # testing_result_after_knn.append(int(new_result / count))
                [max_label, max_count] = max(
                    [[rounds_label, new_result.count(rounds_label)] for rounds_label in list(set(new_result))], key=lambda item: item[1])
            else:
                [max_label, max_count] = [-1, -1]
            self.max_label_count.append([max_label, max_count])
            self.done=self.done+1
            # slow donw by using io,just for show:)
            # print self.done

    def cal_with_least_times(self, least_times):
        testing_result_after_knn = []
        for i in range(len(self.max_label_count)):
            [ml, mc] = self.max_label_count[i]
            if ml == -1 or mc < least_times:
                testing_result_after_knn.append(self.y[i])
            else:
                testing_result_after_knn.append(ml)
        acc = float(sum([1 for i in range(len(self.true_result))
                         if testing_result_after_knn[i] == self.true_result[i]])) / len(self.y)
        return [round(acc*100,2), testing_result_after_knn]

    def getprogress(self):
        # how many have we done
        return float(self.done) / self.total

    def getresult(self):
        # not done yet
        if self.done != self.total:
            return [-1, -1,-1,-1]
        else:
            uniq_ele = list(set(self.y))

            trak = self.traks[self.result.index(max(self.result))]
            # sio.savemat("hahaha.mat",{"trak":trak,"tr":self.true_result,"y":self.y})
            # get testing result' location and label
            y_location = map(lambda x, y: [x, y], self.location, trak)
            # print y_location
            y_location.sort(key=lambda x: x[1])
            trak_location = []
            for i in uniq_ele:
                trak_location.append(
                    map(lambda x: x[0], filter(lambda x: x[1] == i, y_location)))
                trak_location.append([])

            # get testing result' location and label
            y_location = map(lambda x, y: [x, y], self.location, self.y)
            # print y_location
            y_location.sort(key=lambda x: x[1])
            te_location = []
            for i in uniq_ele:
                te_location.append(
                    map(lambda x: x[0], filter(lambda x: x[1] == i, y_location)))
                te_location.append([])

            return [str(range((2 * self.margin + 1) ** 2))[1:-1], str(self.result)[1:-1], te_location, trak_location,str(uniq_ele)[1:-1]]
