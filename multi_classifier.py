import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os
import multi_method
import random
import statistical_feature as sf
import Task
from uuid import uuid4
import time

from tornado.options import define, options
define("port", default=8888, help="run on the given port", type=int)

# save the data user upload
upload_dir = "upload_data/"
download_dir = "result/"


# start page /
class ClassifierHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("classifier.html", test="test")

    def post(self):
        # get data ,return error page if it's empty
        try:
            fileinfo = self.request.files["all_data"][0]
            all_data = fileinfo["body"]
        except KeyError:
            self.render("error.html", error_info="Please Upload Your Data.")
            return

        # save data for calculate
        filename = upload_dir + fileinfo["filename"]

        while os.path.exists(filename):
            filename = filename + str(random.randint(1, 9))

        f = open(filename, "w")
        f.write(all_data)
        f.close()

        try:
            # get method to calculate, default:svm with polynomial kernel
            # method
            method = self.get_argument("method", "SVM")
            kernel = self.get_argument("kernel", "Polynomial")
            C = float(self.get_argument("C", "1"))
            gamma = float(self.get_argument("gamma", "0"))
            selected_feature = self.get_argument("selected_feature", "")
            # overall accuracy,average accuracy,accuracy of each label
            # true data's location of each label,testing result's location of
            # each label,label
            [OA, AA, kappa, label_accuracy, y_te_location, t_te_location, uniq_ele, result_filename] = multi_method.calculate(
                filename, selected_feature, method=method, kernel=kernel, C=C, gamma=gamma)
            # download path
            result_filename = "/" + result_filename

            # for test
            # OA = 59.54
            # AA = 48.62
            # kappa = 52.14
            # label_accuracy = {1: 0.8421052631578947, 2: 0.7097242380261248, 3: 0.6423076923076924, 4: 0.8074866310160428, 5: 0.8614318706697459, 6: 0.8764705882352941, 7: 0.9333333333333333, 8:
            # 0.9579439252336449, 9: 0.8, 10: 0.7505422993492408, 11:
            # 0.5193347193347193, 12: 0.7237569060773481, 13: 0.9612903225806452,
            # 14: 0.8337448559670781, 15: 0.49404761904761907, 16:
            # 0.9565217391304348}
            # uniq_ele=range(1,17,1)

        # remove data after calculating
        except Exception, e:
            print e
            os.remove(filename)
            self.render(
                "error.html", error_info="Please Check Your Data Format.")
            return
        os.remove(filename)

        # error of data's format
        # if (OA < 0):
        #     self.render(
        #         "error.html", error_info="Please Check Your Data Format.")

        self.render("classifier_result.html", method=method, kernel=kernel,
                    C=C, gamma=gamma, OA=OA, AA=AA, kappa=kappa, label_accuracy=label_accuracy, y_te_location=y_te_location, t_te_location=t_te_location, uniq_ele=uniq_ele, result_filename=result_filename)


# data format page /data_format
class DataFormatHandler(tornado.web.RequestHandler):

    def get(self, input):
        if input == "c":
            # format of data for classification
            self.render("c_format.html")
        elif input == "s":
            # format of data for feature statistics
            self.render("s_format.html")
        elif input == "r":
            self.render("r_format.html")
        else:
            self.render("error.html", error_info="Not Found.")


class StatFeatureHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("stat_feature.html")

    def post(self):
        # get data ,return error page if it's empty
        try:
            fileinfo = self.request.files["whole_data"][0]
            whole_data = fileinfo["body"]
        except KeyError:
            self.render("error.html", error_info="Please Upload Your Data.")
            return

        try:
            # save data for calculate
            filename = upload_dir + fileinfo["filename"]

            while os.path.exists(filename):
                filename = filename + str(random.randint(1, 9))

            f = open(filename, "w")
            f.write(whole_data)
            f.close()

            # average of each label,variance of each label
            # label count,label
            [ave, var, y_count, label_location, uniq_ele] = sf.calculate(
                filename)

            # remove data after calculating
        except Exception, e:
            os.remove(filename)
            self.render(
                "error.html", error_info="Please Check Your Data Format.")
            return
        os.remove(filename)

        self.render(
            "stat_feature_result.html", ave=ave, var=var, y_count=y_count, label_location=label_location, uniq_ele=uniq_ele)


class RBFGammaHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("rbf_gamma.html")

    def post(self):
        # get data ,return error page if it's empty
        try:
            fileinfo = self.request.files["rbf_gamma_data"][0]
            rbf_gamma_data = fileinfo["body"]
        except KeyError:
            self.render("error.html", error_info="Please Upload Your Data.")
            return

        try:
            # save data for calculate
            filename = upload_dir + fileinfo["filename"]

            while os.path.exists(filename):
                filename = filename + str(random.randint(1, 9))

            f = open(filename, "w")
            f.write(rbf_gamma_data)
            f.close()

            min_g = float(self.get_argument("gamma_min", "1"))
            max_g = float(self.get_argument("gamma_max", "10"))
            step = float(self.get_argument("gamma_step", "1"))
            C = float(self.get_argument("C", "1"))

            # min_g = 1.0e-8
            # max_g = 2.0e-8
            # step = 1e-9
            # task of calculate with different gamma of RBF
            rgTask = Task.RBFGammaTask(filename, min_g, max_g, step, C)

            # remove data after calculating
        except Exception, e:
            os.remove(filename)
            self.render(
                "error.html", error_info="Please Check Your Data Format or Selected Gammas")
            return
        os.remove(filename)

        # create uniq session of each task
        session = str(uuid4())
        # register the task for requesting
        self.application.taskTable.register(session, rgTask)
        # run task on another thread
        rgTask.start()

        self.render(
            "rbf_gamma_result.html", session=session)


class StateHandler(tornado.web.RequestHandler):

    def get(self, input):
        # get progress of task
        if input == "progress":
            # self.write('{"progress":"%.2f"}' % 1)
            # return
            # get uniq session
            self.session = self.get_argument("session")
            # print self.application.taskTable
            # print self.session
            # get progress of special task
            progress = self.application.taskTable.getprogress(self.session)
            # print progress
            self.write('{"progress":"%.2f"}' % progress)
        # get result of task
        elif input == "result":
            # self.write('{"axis":"%s","acc":"%s"}' % (str(range(20))[1:-1], str(range(20))[1:-1]))
            # return
            # get uniq session
            self.session = self.get_argument("session")
            # get result of special task
            result = self.application.taskTable.getresult(self.session)
            if result[0] != -1:
                # getting result means task have done,so unregister the task
                self.application.taskTable.unregister(self.session)
                # in RBFGamma task,return axis of different gamma,acc of respective
                # gamma
                self.write('{"axis":"%s","acc":"%s"}' % (result[0], result[1]))
        elif input == "knn_result":
            # get uniq session
            self.session = self.get_argument("session")
            # get result of special task
            result = self.application.taskTable.getresult(self.session)
            if result[0] != -1:
                # getting result means task have done,so unregister the task
                self.application.taskTable.unregister(self.session)
                # in RBFGamma task,return axis of different gamma,acc of respective
                # gamma
                self.write('{"axis":"%s","acc":"%s","te_location":"%s","trak_location":"%s","uniq_ele":"%s"}' % (
                    result[0], result[1], result[2], result[3], result[4]))
        else:
            return


class SoftMarginHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("soft_margin.html")

    def post(self):
        # get data ,return error page if it's empty
        try:
            fileinfo = self.request.files["soft_margin_data"][0]
            soft_margin_data = fileinfo["body"]
        except KeyError:
            self.render("error.html", error_info="Please Upload Your Data.")
            return

        try:
            # save data for calculate
            filename = upload_dir + fileinfo["filename"]

            while os.path.exists(filename):
                filename = filename + str(random.randint(1, 9))

            f = open(filename, "w")
            f.write(soft_margin_data)
            f.close()

            min_sf = float(self.get_argument("soft_margin_min", "1"))
            max_sf = float(self.get_argument("soft_margin_max", "10"))
            step = float(self.get_argument("soft_margin_step", "1"))
            kernel = self.get_argument("kernel", "Polynomial")
            gamma = float(self.get_argument("gamma", "0"))

            if min_sf <= 0 or min_sf >= max_sf:
                raise Exception

            # print min_sf
            # print max_sf
            # print step
            # print kernel
            # print gamma

            # min_g = 1.0e-8
            # max_g = 2.0e-8
            # step = 1e-9
            # task of calculate with different gamma of RBF
            sfTask = Task.SoftMarginTask(
                filename, min_sf, max_sf, step, kernel, gamma)

            # remove data after calculating
        except Exception, e:
            os.remove(filename)
            self.render(
                "error.html", error_info="Please Check Your Data Format or Selected Soft Margins")
            return
        os.remove(filename)

        # create uniq session of each task
        session = str(uuid4())
        # register the task for requesting
        self.application.taskTable.register(session, sfTask)
        # run task on another thread
        sfTask.start()

        self.render(
            "soft_margin_result.html", kernel=kernel, gamma=gamma, session=session)


class ForwardStepwiseHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("forward_stepwise.html")

    def post(self):
        # get data ,return error page if it's empty
        try:
            fileinfo = self.request.files["forward_stepwise_data"][0]
            forward_stepwise_data = fileinfo["body"]
        except KeyError:
            self.render("error.html", error_info="Please Upload Your Data.")
            return

        try:
            # save data for calculate
            filename = upload_dir + fileinfo["filename"]

            while os.path.exists(filename):
                filename = filename + str(random.randint(1, 9))

            f = open(filename, "w")
            f.write(forward_stepwise_data)
            f.close()

            n_features = int(self.get_argument("number_of_features", "1"))

            if n_features <= 0:
                raise Exception

            fsTask = Task.ForwardStepwiseTask(filename, n_features)

            # remove data after calculating
        except Exception, e:
            print e
            os.remove(filename)
            self.render(
                "error.html", error_info="Please Check Your Data Format or Selected Soft Margins")
            return
        os.remove(filename)

        # create uniq session of each task
        session = str(uuid4())
        # register the task for requesting
        self.application.taskTable.register(session, fsTask)
        # run task on another thread
        fsTask.start()

        self.render(
            "forward_stepwise_result.html", session=session)


class KnnHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("knn.html", test="test")

    def post(self):
        # get data ,return error page if it's empty
        try:
            fileinfo = self.request.files["knn_data"][0]
            all_data = fileinfo["body"]
        except KeyError:
            self.render("error.html", error_info="Please Upload Your Data.")
            return

        # save data for calculate
        filename = upload_dir + fileinfo["filename"]

        while os.path.exists(filename):
            filename = filename + str(random.randint(1, 9))

        f = open(filename, "w")
        f.write(all_data)
        f.close()

        try:
            margin = int(self.get_argument("margin", "1"))
            knnTask = Task.KnnTask(filename, margin)
        except Exception, e:
            os.remove(filename)
            self.render(
                "error.html", error_info="Please Check Your Data Format.")
            return
        os.remove(filename)

        # create uniq session of each task
        session = str(uuid4())
        # register the task for requesting
        self.application.taskTable.register(session, knnTask)
        # run task on another thread

        self.render(
            "knn_result.html", session=session)
        knnTask.start()


class ResultDataHandler(tornado.web.RequestHandler):

    def get(self, filename, suffix):
        try:
            filename = filename + "." + suffix
            f = open(download_dir + filename)
            self.set_header('Content-Type', 'application/octet-stream')
            self.set_header(
                'Content-Disposition', 'attachment; filename=' + 'result.mat')
            self.write(f.read())
            self.finish()
        except Exception, e:
            self.render("error.html", error_info="File\'s not found.")


class DeleteHandler(tornado.websocket.WebSocketHandler):

    def open(self):
        self.filename = self.get_argument("filename")[1:]

    def on_close(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def on_message(self, message):
        pass


class Application(tornado.web.Application):

    def __init__(self):
        # get tasktable for administrating tasks
        self.taskTable = Task.TaskTable()

        handlers = [
            (r"/", ClassifierHandler),
            (r"/format/(\w+)", DataFormatHandler),
            (r"/result/(\w+).(\w+)", ResultDataHandler),
            (r"/delete", DeleteHandler),
            (r"/stat_feature", StatFeatureHandler),
            (r"/rbf_gamma", RBFGammaHandler),
            (r"/soft_margin", SoftMarginHandler),
            (r"/forward_stepwise", ForwardStepwiseHandler),
            (r"/state/(\w+)", StateHandler),
            (r"/knn", KnnHandler)
        ]

        settings = {
            'template_path': 'templates',
            'static_path': 'static',
            'debug': 'True'
        }

        tornado.web.Application.__init__(self, handlers, **settings)


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(8989)
    tornado.ioloop.IOLoop.instance().start()
