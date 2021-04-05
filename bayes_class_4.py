from __future__ import division
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import math

# ROS topics values are inserted as text files (data4 text files)
Path1 = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path1_20.text', delimiter=',', usecols=(1), unpack=True)
Path2 = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path2_20.text', delimiter=',', usecols=(1), unpack=True)
Path3 = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path3_20.text', delimiter=',', usecols=(1), unpack=True)
Path4 = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path4_20.text', delimiter=',', usecols=(1), unpack=True)
Path5 = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path5_20.text', delimiter=',', usecols=(1), unpack=True)
Angle1 = np.loadtxt('/home/dimiubuntu/data_collection/data4/angle/angle1_20.text', delimiter=',', usecols=(1), unpack=True)
Angle2 = np.loadtxt('/home/dimiubuntu/data_collection/data4/angle/angle2_20.text', delimiter=',', usecols=(1), unpack=True)
Angle3 = np.loadtxt('/home/dimiubuntu/data_collection/data4/angle/angle3_20.text', delimiter=',', usecols=(1), unpack=True)
Angle4 = np.loadtxt('/home/dimiubuntu/data_collection/data4/angle/angle4_20.text', delimiter=',', usecols=(1), unpack=True)
Angle5 = np.loadtxt('/home/dimiubuntu/data_collection/data4/angle/angle5_20.text', delimiter=',', usecols=(1), unpack=True)
Dis1 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis1_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Dis2 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis2_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Dis3 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis3_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Dis4 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis4_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Dis5 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis5_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Term1 = np.loadtxt('/home/dimiubuntu/data_collection/data4/term/term1_20.text', delimiter=',', usecols=(1), unpack=True)
Term2 = np.loadtxt('/home/dimiubuntu/data_collection/data4/term/term2_20.text', delimiter=',', usecols=(1), unpack=True)
Term3 = np.loadtxt('/home/dimiubuntu/data_collection/data4/term/term3_20.text', delimiter=',', usecols=(1), unpack=True)
Term4 = np.loadtxt('/home/dimiubuntu/data_collection/data4/term/term4_20.text', delimiter=',', usecols=(1), unpack=True)
Term5 = np.loadtxt('/home/dimiubuntu/data_collection/data4/term/term5_20.text', delimiter=',', usecols=(1), unpack=True)

iter = len(Path1)

class Bayes:
    def __init__(self, n, Delta, Angle, wA, Path, wP, maxA, maxP, Dis):
        self.n = n
        self.Delta = Delta
        self.Angle = Angle
        self.wA = wA
        self.Path = Path
        self.wP = wP
        self.maxA = maxA
        self.maxP = maxP
        self.Dis = Dis

    def initialization_prior(self):
        prior = np.ones(self.n) * (1 / self.n)
        return prior

    def cpt(self):
        data1 = np.ones((self.n, self.n)) * (self.Delta / (self.n - 1))
        np.fill_diagonal(data1, 1 - self.Delta)
        cpt = data1
        return cpt

    def compute_like(self):
        like = np.exp(-9*self.Dis)
        return like

    # def compute_like(self):  # sensor model .. alternatives at the end of the script
    #     a = self.Angle / np.sum(self.Angle)
    #     p = self.Path / np.sum(self.Path)
    #     final = np.exp(-a/self.wA) * np.exp(-p/self.wP)
    #     return final

    # def compute_like(self):
    #     a = self.Angle / self.maxA
    #     print(a)
    #     p = self.Path / self.maxP
    #     print(p)
    #     like = np.exp(-a / self.wA) * np.exp(-p / self.wP)
    #     return like

    def compute_conditional(self):
        out1 = np.matmul(cpt, prior.T)
        sum = out1
        return sum

    def compute_post(self):
        out2 = likelihood * summary
        post = out2 / np.sum(out2)
        return post

total = np.array([])
post_max = np.array([])
posterior_1 = np.array([])
posterior_2 = np.array([])
posterior_3 = np.array([])
posterior_4 = np.array([])
posterior_5 = np.array([])
posterior = []
i = 0
while i < iter:
    Angle = np.array([Angle1[i], Angle2[i], Angle3[i], Angle4[i], Angle5[i]])
    Path = np.array([Path1[i], Path2[i], Path3[i], Path4[i], Path5[i]])
    Dis = np.array([Dis1[i], Dis2[i], Dis3[i], Dis4[i], Dis5[i]])        # give me the sensor distance measurements as they have been logged in text files (for RBII & Carlson)
    # print("Angle : " +str(Angle), "Path : " + str(Path))
    bayes = Bayes(5, 0.2, Angle, 0.6, Path, 0.4, 180, 25, Dis)
    cpt = bayes.cpt()
    if i == 0:
        prior = bayes.initialization_prior()
    else:
        prior = posterior
    print("prior is :" +str(prior))
    likelihood = bayes.compute_like()
    summary = bayes.compute_conditional()
    posterior = bayes.compute_post()
    value = np.amax(posterior)
    post_max = np.append(post_max, value)
    posterior_1 = np.append(posterior_1, posterior[0])
    posterior_2 = np.append(posterior_2, posterior[1])
    posterior_3 = np.append(posterior_3, posterior[2])
    posterior_4 = np.append(posterior_4, posterior[3])
    posterior_5 = np.append(posterior_5, posterior[4])
    index = np.argmax(posterior)
    total = np.append(total, index + 1)

    print("iteration: " + str(i), "POSTERIOR: " + str(posterior), "most probable goal is: " + str(index + 1))
    print("----------------------------------------------------------------------------------------------------")
    i = i + 1

print("total is : " +str(total))
print("MAX is : " +str(post_max))

# EVALUATION
# after bayes results make evaluation properly .. find the "corrects" on each group
def evaluation(iter, total):
    indic = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path1_20.text', delimiter=',', usecols=(0), unpack=True) # we use the first column of any of the extracted text files
    change = np.loadtxt('/home/dimiubuntu/data_collection/data4/change/change_20.text', delimiter=',', usecols=(0),
                        unpack=True) # ROS time that intent was changed (from text file)

    print("number of intent changes : " + str(
        len([i for i in indic if i in change])))  # how many changes of intent happened  (HERE only 1 change)
    print("ROS change time : " + str([i for i in indic if
                                      i in change]))  # confirm that changes (ROS time values) exist based on any of the extracted text files

    # find the position of where the change is located
    position = [i for i, value in enumerate(indic, start=0) if value == change]
    print(position)
    print(type(position))

    # run through total array and find correct on each group
    # values until first change --> find correct1 (here actual goal = 2)
    total1 = np.array([])
    for i in range(position[0]):  # new[0]-1
        total1 = np.append(total1, total[i])
    print("total1 : " + str(total1))
    correct1 = np.count_nonzero(total1 == 5)  # 4 .. 2 .. 2 .. 5 .. 2 .. 1 .. 1 .. 2 .. 2 .. 5 .. 4 .. 2 .. 1 .. 1 .. 1
    print("No of total1 measurements : " + str(len(total1)), "correct1 : " + str(correct1))
    print("------------------------------------------------------------------------------------------------------")

    # values from first "change" until end --> find correct2 (here actual goal = 1)
    total2 = np.array([])
    for i in range(position[0], iter):  # new[0]-1, iter
        total2 = np.append(total2, total[i])
    print("total2 : " + str(total2))
    correct2 = np.count_nonzero(total2 == 1)  # 3 .. 5 .. 3 .. 4 .. 4 .. 5 .. 2 .. 5 .. 3 .. 3 .. 1 .. 5 .. 5 .. 3 .. 2
    print("No of total2 measurements : " + str(len(total2)), "correct2 : " + str(correct2))
    print("------------------------------------------------------------------------------------------------------")

    Correct = correct1 + correct2  # sum of corrects
    success = ("{0:.2%}".format(Correct / iter))  # find the success rate (Correct/iterations)
    print("% of success: " + str(success))

    return success, position, total1, total2


# evaluate the "jitter" metric .. once the algorithm has captured the correct goal after change check if it will lose it till the end
def jitter_evaluation(total1, total2):
    value_correct1 = 5
    ind1 = np.where(total1 == value_correct1)[0]
    first_occurrence1 = ind1[0]
    print("first occurrence of correct=2 after change in", first_occurrence1)
    fail1 = len([i for i in total1[first_occurrence1:] if i != value_correct1])
    jitter1 = fail1 / (len(total1) - first_occurrence1)
    jitter1_perce = ("{0:.2%}".format(jitter1))
    print("how many fails1=",fail1, "," "jitter1=",jitter1_perce)

    value_correct2 = 1
    ind2 = np.where(total2 == value_correct2)[0]
    first_occurrence2 = ind2[0]
    print("first occurrence of correct=1 after change in", first_occurrence2)
    fail2 = len([i for i in total2[first_occurrence2:] if i != value_correct2])
    jitter2 = fail2 / (len(total2) - first_occurrence2)
    jitter2_perce = ("{0:.2%}".format(jitter2))
    print("how many fails2=",fail2, "," "jitter2=",jitter2_perce)

    fail = fail1 + fail2
    JITTER = ("{0:.2%}".format(jitter1 + jitter2))
    print("total FAILS=",fail, "," "total JITTER=",JITTER)

    return fail, JITTER


# Cross entropy (log-loss)
def log_loss():
    loss = np.array([])
    y_real1 = [1, 0, 0, 0, 0]  # when goal 1 is the actual
    y_real2 = [0, 1, 0, 0, 0]  # when goal 2 is the actual
    y_real3 = [0, 0, 1, 0, 0]  # when goal 3 is the actual
    y_real4 = [0, 0, 0, 1, 0]  # when goal 4 is the actual
    y_real5 = [0, 0, 0, 0, 1]  # when goal 5 is the actual
    for i in range(iter):
        if i < position[0]:
            #loss = np.append(loss, (-y_real1[0] * np.log10(posterior_1[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real2[1] * np.log10(posterior_2[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real3[2] * np.log10(posterior_3[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real4[3] * np.log10(posterior_4[i])))  # simplified log-loss formula
            loss = np.append(loss, (-y_real5[4] * np.log10(posterior_5[i])))  # simplified log-loss formula
        else:
            loss = np.append(loss, (-y_real1[0] * np.log10(posterior_1[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real2[1] * np.log10(posterior_2[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real3[2] * np.log10(posterior_3[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real4[3] * np.log10(posterior_4[i])))  # simplified log-loss formula
            #loss = np.append(loss, (-y_real5[4] * np.log10(posterior_5[i])))  # simplified log-loss formula
    log_loss = np.mean(loss)
    print("Log Loss = ", log_loss)
    return log_loss

results, position, total1, total2 = evaluation(iter, total)
log_loss = log_loss()
fail, JITTER = jitter_evaluation(total1, total2)


# ------------------------------------ END (scenario 4) ------------------------------------
