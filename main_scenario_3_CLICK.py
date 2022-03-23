from __future__ import division
import numpy as np
from BayesianOperatorIntentRecognition_AIRM import BayesianOperatorIntentRecognition_AIRM

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
    # ROS topics values are inserted as text files (data3 text files)
    indic = np.loadtxt('/home/dimiubuntu/data_collection/data3/path/path1_20.text', delimiter=',', usecols=(0),
                       unpack=True)
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

    iter = len(Angle1)
    print("iterations=", iter)

    number_of_goals = 3
    DELTA = 0.2
    WEIGHT_ANGLE = 0.6
    WEIGHT_PATH = 0.4
    LIMIT_ANGLE = 180
    LIMIT_PATH = 25
    flag = 0
    time = 10
    P0 = 0.95  # goal's probability when click is active
    threshold = 0.35
    rest = (1 - P0) / (number_of_goals - 1)  # probability of the remaining goals (these that are not being clicked)

    CLICKS = np.loadtxt('/home/dimiubuntu/data_collection/data3/click/click_20.text', delimiter=',', usecols=(0),
                        unpack=True)
    print("number of clicks : ", len([i for i in indic if i in CLICKS]))  # how many clicks
    print("ROS click time : ", [i for i in indic if
                                i in CLICKS])  # confirm that clicks (ROS time values) exist based on any of the extracted text files

    # find the positions where the clicks happened
    Clicks = np.array([])
    for k in CLICKS:
        CLICKS = [i for i, value in enumerate(indic, start=0) if value == k]
        Clicks = np.append(Clicks, CLICKS)  # float array
    click = Clicks.astype(np.int64)  # convert to int array because we need it as index parameter
    print("The clicks are located in positions : ", click)

    # our MAIN loop
    total = np.array([])  # array for saving the max indices (~which goal is most probable)
    posterior_1 = np.array([])
    posterior_2 = np.array([])
    posterior_3 = np.array([])
    prior = np.array([])
    for i in range(len(indic)):
        Angle = np.array([Angle1[i], Angle2[i], Angle3[i]])
        Path = np.array([Path1[i], Path2[i], Path3[i]])
        if (i < time) and (i == click[0]) and (flag == 0):  # set prior when first click is active
            prior = np.array([P0, rest, rest])
            flag = 1
        elif (i >= click[1]) and (i < click[1] + time) and (flag == 1):  # set prior when second click is active
            prior = np.array([rest, P0, rest])
            flag = 2
        elif (i >= click[2]) and (i < click[2] + time) and (flag == 2):  # set prior when third click is active
            prior = np.array([rest, rest, P0])
            flag = 3

        bayes = BayesianOperatorIntentRecognition_AIRM(number_of_goals, DELTA, Angle, WEIGHT_ANGLE, Path, WEIGHT_PATH,
                                                       LIMIT_ANGLE, LIMIT_PATH, P0, threshold, time, i)

        if i < time:
            likelihood = bayes.compute_like()
            summary = bayes.compute_conditional(prior)
            dec = bayes.compute_decay_scenario_3(i, click)
            plus = bayes.extra_term(summary, dec)
            posterior = bayes.compute_final(likelihood, plus)
            index = bayes.get_maximium_value(posterior)
            value_max = np.amax(posterior)

            print(
                "decay", "iter=", i, "MAX:", index + 1, "prior:", prior, "posterior:", posterior,
                "decay:", dec)
            total = np.append(total, index + 1)
            prior = posterior

        elif (i >= click[1]) and (i < click[1] + time):
            likelihood = bayes.compute_like()
            summary = bayes.compute_conditional(prior)
            dec = bayes.compute_decay_scenario_3(i - click[1], click)
            plus = bayes.extra_term(summary, dec)
            posterior = bayes.compute_final(likelihood, plus)
            index = bayes.get_maximium_value(posterior)
            value_max = np.amax(posterior)

            print(
                "decay", "iter=", i, "MAX:", index + 1, "prior:", prior, "posterior:", posterior,
                "decay:", dec)
            total = np.append(total, index + 1)
            prior = posterior

        elif (i >= click[2]) and (i < click[2] + time):
            likelihood = bayes.compute_like()
            summary = bayes.compute_conditional(prior)
            dec = bayes.compute_decay_scenario_3(i - click[2], click)
            plus = bayes.extra_term(summary, dec)
            posterior = bayes.compute_final(likelihood, plus)
            index = bayes.get_maximium_value(posterior)
            value_max = np.amax(posterior)

            print(
                "decay", "iter=", i, "MAX:", index + 1, "prior:", prior, "posterior:", posterior,
                "decay:", dec)
            total = np.append(total, index + 1)
            prior = posterior

        else:
            likelihood = bayes.compute_like()
            summary = bayes.compute_conditional(prior)
            posterior = bayes.compute_post(likelihood, summary)
            index = bayes.get_maximium_value(posterior)
            value_max = np.amax(posterior)

            print(
                "normal", "iter=", i, "MAX:", index + 1, "prior:", prior, "posterior:", posterior)
            total = np.append(total, index + 1)
            prior = posterior

        posterior_1 = np.append(posterior_1, posterior[0])
        posterior_2 = np.append(posterior_2, posterior[1])
        posterior_3 = np.append(posterior_3, posterior[2])

    print(total)


    # algorithm's EVALUATION
    # after bayes results make evaluation properly .. find the "corrects" on each group
    def evaluation(iter, total):
        total1 = np.array([])
        for i in range(click[1]):
            total1 = np.append(total1, total[i])
        print("total1 : ", total1)
        correct1 = np.count_nonzero(total1 == 1)
        print("No of total1 measurements = ", len(total1), "correct1 = ", correct1)
        print("------------------------------------------------------------------------------------------------------")

        # values from first "change" until second "change" --> find correct2 (here actual goal = 2)
        total2 = np.array([])
        for i in range(click[1], click[2]):
            total2 = np.append(total2, total[i])
        print("total2 : ", total2)
        correct2 = np.count_nonzero(total2 == 2)
        print("No of total2 measurements = ", len(total2), "correct2 = ", correct2)
        print("------------------------------------------------------------------------------------------------------")

        # values from second "change" until end --> find correct3 (here actual goal = 3)
        total3 = np.array([])
        for i in range(click[2], iter):
            total3 = np.append(total3, total[i])
        print("total3 : ", total3)
        correct3 = np.count_nonzero(total3 == 3)
        print("No of total3 measurements = ", len(total3), "correct3 = ", correct3)
        print("------------------------------------------------------------------------------------------------------")

        Correct = correct1 + correct2 + correct3  # sum of corrects
        success = ("{0:.2%}".format(Correct / iter))  # find the success rate (Correct/iterations)
        print("% of success : ", success)

        return success, click, total1, total2, total3


    # Cross entropy (log-loss)
    def log_loss():
        loss = np.array([])
        y_real1 = [1, 0, 0]  # when goal 1 is the actual
        y_real2 = [0, 1, 0]  # when goal 2 is the actual
        y_real3 = [0, 0, 1]  # when goal 3 is the actual
        for i in range(iter):
            if i < click[1]:
                loss = np.append(loss, (-y_real1[0] * np.log10(posterior_1[i])))  # simplified log-loss formula
            elif i >= click[1] and i < click[2]:
                loss = np.append(loss, (-y_real2[1] * np.log10(posterior_2[i])))  # simplified log-loss formula
            else:
                loss = np.append(loss, (-y_real3[2] * np.log10(posterior_3[i])))  # simplified log-loss formula
        log_loss = np.mean(loss)
        print("Log Loss = ", log_loss)
        return log_loss


    results, click, total1, total2, total3 = evaluation(iter, total)
    log_loss = log_loss()


