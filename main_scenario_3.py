from __future__ import division
import numpy as np
from BayesianOperatorIntentRecognition import BayesianOperatorIntentRecognition

"""
Copyright (c) 2020-2021, Dimitris Panagopoulos
All rights reserved.
----------------------------------------------------
"""
# Short Description of this script : Instead of running trials all over again in ROS platform, this script encourages the
# user to test different/modified algorithms (based on Bayesian Estimation).. Crucial ROS topics values have been extracted
# (i.e. Path, Angle, Time model, etc) in the form of text files .. Once they are introduced HERE, each trial can be generated under
# conditions may seem practically different (e.g. modified observation model) .. The script does the same job similar to ROS (in
# the context of bayesian system's evaluation) based on experiments that have already been executed, just by utilizing the
# generated topic values.

if __name__ == '__main__':
    Path1 = np.loadtxt('/home/dimiubuntu/data_collection/data3/path/path1_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Path2 = np.loadtxt('/home/dimiubuntu/data_collection/data3/path/path2_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Path3 = np.loadtxt('/home/dimiubuntu/data_collection/data3/path/path3_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Angle1 = np.loadtxt('/home/dimiubuntu/data_collection/data3/angle/angle1_20.text', delimiter=',', usecols=(1),
                        unpack=True)
    Angle2 = np.loadtxt('/home/dimiubuntu/data_collection/data3/angle/angle2_20.text', delimiter=',', usecols=(1),
                        unpack=True)
    Angle3 = np.loadtxt('/home/dimiubuntu/data_collection/data3/angle/angle3_20.text', delimiter=',', usecols=(1),
                        unpack=True)
    Dis1 = np.loadtxt('/home/dimiubuntu/data_collection/data3/euclidean/dis1_20.text', delimiter=',', usecols=(1),
                      unpack=True)  # for RBII & Carlson
    Dis2 = np.loadtxt('/home/dimiubuntu/data_collection/data3/euclidean/dis2_20.text', delimiter=',', usecols=(1),
                      unpack=True)  # for RBII & Carlson
    Dis3 = np.loadtxt('/home/dimiubuntu/data_collection/data3/euclidean/dis3_20.text', delimiter=',', usecols=(1),
                      unpack=True)  # for RBII & Carlson
    iter = len(Path1)

    number_of_goals = 3
    DELTA = 0.2
    WEIGHT_ANGLE = 0.6
    WEIGHT_PATH = 0.4
    LIMIT_ANGLE = 180
    LIMIT_PATH = 25
    total = np.array([])
    posterior_1 = np.array([])
    posterior_2 = np.array([])
    posterior_3 = np.array([])
    posterior = []
    i = 0

    while i < iter:
        Angle = np.array([Angle1[i], Angle2[i], Angle3[i]])
        Path = np.array([Path1[i], Path2[i], Path3[i]])
        Dis = np.array([Dis1[i], Dis2[i], Dis3[
            i]])  # give me the sensor distance measurements as they have been logged in text files (for RBII & Carlson)
        bayes = BayesianOperatorIntentRecognition(number_of_goals, DELTA, Angle, WEIGHT_ANGLE, Path, WEIGHT_PATH,
                                                  LIMIT_ANGLE, LIMIT_PATH, Dis, i)
        if i == 0:
            prior = bayes.initialization_prior()
        else:
            prior = posterior
        print("prior is :", prior)
        likelihood = bayes.compute_like()
        summary = bayes.compute_conditional(prior)
        posterior = bayes.compute_post(likelihood, summary)
        index = bayes.get_maximium_value(posterior)
        value = np.amax(posterior)

        posterior_1 = np.append(posterior_1, posterior[0])
        posterior_2 = np.append(posterior_2, posterior[1])
        posterior_3 = np.append(posterior_3, posterior[2])

        total = np.append(total, index + 1)

        print("iteration: ", i, "---", "POSTERIOR: ", posterior, "---", "most probable goal is: ", index + 1)
        print("----------------------------------------------------------------------------------------------------")
        i = i + 1

    print("total is : ", total)


    # algorithm's EVALUATION
    # after bayes results make evaluation properly .. find the "corrects" on each group
    def evaluation(iter, total):
        indic = np.loadtxt('/home/dimiubuntu/data_collection/data3/path/path1_20.text', delimiter=',', usecols=(0),
                           unpack=True)  # we use the first column of the extracted text files
        change = np.loadtxt('/home/dimiubuntu/data_collection/data3/change/change_20.text', delimiter=',', usecols=(0),
                            unpack=True)  # ROS time that intent was changed (from text file)  (HERE 2 changes)
        print("number of intent changes : ",
              len([i for i in indic if i in change]))  # how many changes of intent happened  (HERE 2 changes)
        print("time values of intent change : ", [i for i in indic if
                                                  i in change])  # confirm that changes (ROS time values) exist based on any of the extracted text files

        # find the positions of where the changes are located
        Changes = np.array([])
        for k in change:
            changes = [i for i, value in enumerate(indic, start=0) if value == k]
            Changes = np.append(Changes, changes)  # float array
        position = Changes.astype(np.int64)  # convert to int array
        print("The changes of intent are located in positions : ", position)  # int array
        print("------------------------------------------------------------------------------------------------------")

        # run through total array and find correct on each group
        # values until first change --> find correct1 (here actual goal = 1)
        total1 = np.array([])
        for i in range(position[0]):
            total1 = np.append(total1, total[i])
        print("total1 : ", total1)
        correct1 = np.count_nonzero(total1 == 1)
        print("No of total1 measurements : ", len(total1), "correct1 : ", correct1)
        print("------------------------------------------------------------------------------------------------------")

        # values from first "change" until second "change" --> find correct2 (here actual goal = 2)
        total2 = np.array([])
        for i in range(position[0], position[1]):
            total2 = np.append(total2, total[i])
        print("total2 : ", total2)
        correct2 = np.count_nonzero(total2 == 2)
        print("No of total2 measurements : ", len(total2), "correct2 : ", correct2)
        print("------------------------------------------------------------------------------------------------------")

        # values from second "change" until end --> find correct3 (here actual goal = 3)
        total3 = np.array([])
        for i in range(position[1], iter):  # new[1]-1, iter
            total3 = np.append(total3, total[i])
        print("total3 : ", total3)
        correct3 = np.count_nonzero(total3 == 3)
        print("No of total3 measurements : ", len(total3), "correct3 : ", correct3)
        print("------------------------------------------------------------------------------------------------------")

        Correct = correct1 + correct2 + correct3  # sum of corrects
        success = ("{0:.2%}".format(Correct / iter))  # find the success rate (Correct/iterations)
        print("% of success : ", success)

        return success, position, total1, total2, total3


    # Cross entropy (log-loss)
    def log_loss():
        loss = np.array([])
        y_real1 = [1, 0, 0]  # when goal 1 is the actual
        y_real2 = [0, 1, 0]  # when goal 2 is the actual
        y_real3 = [0, 0, 1]  # when goal 3 is the actual
        for i in range(iter):
            if i < position[0]:
                loss = np.append(loss, (-y_real1[0] * np.log10(posterior_1[i])))  # simplified log-loss formula
            elif i >= position[0] and i < position[1]:
                loss = np.append(loss, (-y_real2[1] * np.log10(posterior_2[i])))  # simplified log-loss formula
            else:
                loss = np.append(loss, (-y_real3[2] * np.log10(posterior_3[i])))  # simplified log-loss formula
        log_loss = np.mean(loss)
        print("Log Loss = ", log_loss)
        return log_loss


    # call the evaluation functions
    results, position, total1, total2, total3 = evaluation(iter, total)
    log_loss = log_loss()



