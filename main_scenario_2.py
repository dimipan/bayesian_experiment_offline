from __future__ import division
import numpy as np
from BayesianOperatorIntentRecognition import BayesianOperatorIntentRecognition

"""
Copyright (c) 2020-2021, Dimitris Panagopoulos
All rights reserved.
----------------------------------------------------
"""

if __name__ == '__main__':
    Path1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/path/path1_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Path2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/path/path2_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Path3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/path/path3_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Angle1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/angle/angle1_20.text', delimiter=',', usecols=(1),
                        unpack=True)
    Angle2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/angle/angle2_20.text', delimiter=',', usecols=(1),
                        unpack=True)
    Angle3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/angle/angle3_20.text', delimiter=',', usecols=(1),
                        unpack=True)
    Dis1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/euclidean/dis1_20.text', delimiter=',', usecols=(1),
                      unpack=True)  # for RBII & Carlson
    Dis2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/euclidean/dis2_20.text', delimiter=',', usecols=(1),
                      unpack=True)  # for RBII & Carlson
    Dis3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/euclidean/dis3_20.text', delimiter=',', usecols=(1),
                      unpack=True)  # for RBII & Carlson
    Term1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/term/term1_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Term2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/term/term2_20.text', delimiter=',', usecols=(1),
                       unpack=True)
    Term3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/term/term3_20.text', delimiter=',', usecols=(1),
                       unpack=True)

    iter = len(Path1)
    print(iter)

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
        print("Angle : ", Angle, "Path : ", Path)
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
        i += 1

    print("total is : ", total)


    # # EVALUATION
    # # after bayes results make evaluation properly
    def evaluation(iter, total):
        correct = np.count_nonzero(total == 3)  # find how many correct measurements exist (here actual goal = 3)
        print("correct measurements : ", correct)
        success = ("{0:.2%}".format(correct / (iter)))  # find the success rate (correct/iterations)
        print("% of success : ", success)

        return success


    # Cross entropy (log-loss)
    def log_loss():
        loss = np.array([])
        y_real3 = [0, 0, 1]  # when goal 3 is the actual
        for i in range(iter):
            # loss = np.append(loss, -(y_real3[0]*np.log10(posterior_1[i]) + y_real3[1]*np.log10(posterior_2[i]) + y_real3[2]*np.log10(posterior_3[i])))  # full log-loss formula
            loss = np.append(loss, (-y_real3[2] * np.log10(posterior_3[i])))  # simplified log-loss formula
        log_loss = np.mean(loss)
        print("Log Loss = ", log_loss)
        return log_loss


    results = evaluation(iter, total)
    log_loss = log_loss()






