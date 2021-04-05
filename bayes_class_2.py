from __future__ import division
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import math

Path1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/path/path1_20.text', delimiter=',', usecols=(1), unpack=True)
Path2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/path/path2_20.text', delimiter=',', usecols=(1), unpack=True)
Path3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/path/path3_20.text', delimiter=',', usecols=(1), unpack=True)
Angle1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/angle/angle1_20.text', delimiter=',', usecols=(1), unpack=True)
Angle2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/angle/angle2_20.text', delimiter=',', usecols=(1), unpack=True)
Angle3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/angle/angle3_20.text', delimiter=',', usecols=(1), unpack=True)
Dis1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/euclidean/dis1_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Dis2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/euclidean/dis2_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Dis3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/euclidean/dis3_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
Term1 = np.loadtxt('/home/dimiubuntu/data_collection/data2/term/term1_20.text', delimiter=',', usecols=(1), unpack=True)
Term2 = np.loadtxt('/home/dimiubuntu/data_collection/data2/term/term2_20.text', delimiter=',', usecols=(1), unpack=True)
Term3 = np.loadtxt('/home/dimiubuntu/data_collection/data2/term/term3_20.text', delimiter=',', usecols=(1), unpack=True)
iter = len(Path1)
print(iter)
print(len(Term1))

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

    # def compute_like(self):
    #     like = np.exp(-5.5*self.Dis)
    #     return like


    # def compute_like(self):
    #     a = self.Angle / np.sum(self.Angle)
    #     p = self.Path / np.sum(self.Path)
    #     final = np.exp(-a/self.wA) * np.exp(-p/self.wP)
    #     return final


    def compute_like(self):
        a = self.Angle / self.maxA
        p = self.Path / self.maxP
        like = np.exp(-a / self.wA) * np.exp(-p / self.wP)
        return like

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
i = 0
posterior = []
while i < iter:
    Angle = np.array([Angle1[i], Angle2[i], Angle3[i]])
    Path = np.array([Path1[i], Path2[i], Path3[i]])
    print("Angle : " +str(Angle), "Path : " + str(Path))
    Dis = np.array([Dis1[i], Dis2[i], Dis3[i]])        # give me the sensor distance measurements as they have been logged in text files (for RBII & Carlson)
    bayes = Bayes(3, 0.2, Angle, 0.6, Path, 0.4, 180, 25, Dis)
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
    index = np.argmax(posterior)
    total = np.append(total, index + 1)

    print("iteration: " + str(i), "POSTERIOR: " + str(posterior), "most probable goal is: " + str(index + 1))
    print("----------------------------------------------------------------------------------------------------")
    i = i + 1

print("total is : " +str(total))
print("MAX is : " +str(post_max))
print("POST1", posterior_1)
print("POST2", posterior_2)
print("POST3", posterior_3)
POSTERIOR = np.array([posterior_1, posterior_2, posterior_3])
print("----------------------------------------------------------------------------------------------------")


# # EVALUATION
# # after bayes results make evaluation properly
def evaluation(iter, total, post_max):
    correct = np.count_nonzero(total == 3)  # find how many correct measurements exist (here actual goal = 3)
    print("correct measurements : " + str(correct))
    success = ("{0:.2%}".format(correct / (iter)))  # find the success rate (correct/iterations)
    print("% of success : " + str(success))
    average_perce = (
        "{0:.3%}".format(np.sum(post_max) / len(post_max)))  # find the average percentage value of the max values
    print("average % of max values : " + str(average_perce))

    return success


# evaluate the "jitter" metric .. once the algorithm has captured the correct goal after change, check if it will lose it till the end
def jitter_evaluation(total):
    value_correct = 3
    ind = np.where(total == value_correct)[0]
    first_occurrence = ind[0]
    print("first occurrence of correct=3 after change in", first_occurrence)
    fail = len([i for i in total[first_occurrence:] if i != value_correct])
    jitter = fail / (len(total) - first_occurrence)
    jitter_perce = ("{0:.2%}".format(jitter))
    print("how many fails=",fail, "," "jitter=",jitter_perce)

    return fail, jitter_perce, first_occurrence

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

# # find the difference between most probable goal and 2nd most probable during all the experiment
# def confidence_total():
#     diafora = []
#     for i in range(iter):
#         apotelesma = POSTERIOR[:, i]
#         maximum = np.max(apotelesma)
#         second = np.unique(apotelesma)[-2]
#         #print(maximum, second)
#         difference = maximum - second
#         diafora = np.append(diafora, difference)
#     #print(diafora)
#     print(np.mean(diafora))
# 
# # find the difference between most probable goal and 2nd most probable IFF the algorithm has predicted correctly
# def confidence_correct():
#     diafora = []
#     for i in range(iter):
#         apotelesma = POSTERIOR[:, i]
#         maximum = np.max(apotelesma)
#         second = np.unique(apotelesma)[-2]
#         difference = maximum - second
#         if i >= first_occurrence and i < iter:
#             #print(maximum, second, i)
#             diafora = np.append(diafora, difference)
#         else:
#             print("eimai sto field pou o algorithmos kanei lathos prediction", i)
#             pass
#     #print(diafora)
#     print(np.mean(diafora))


results = evaluation(iter, total, post_max)
log_loss = log_loss()
fail, JITTER, first_occurrence = jitter_evaluation(total)
# confidence_total = confidence_total()
# CONFIDENCE_WHEN_CORRECT = confidence_correct()



