from __future__ import division
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import math
import time
start_time = time.time()

from scipy.spatial import distance

# Short Description of this script : Instead of running trials all over again in ROS platform, this script encourages the
# user to test different/modified algorithms (based on Bayesian Estimation).. Crucial ROS topics values have been extracted
# (i.e. Path, Angle, Time model, etc) in the form of text files .. Once they are introduced HERE, each trial can be generated under
# conditions may seem practically different (e.g. modified observation model) .. The script does the same job similar to ROS (in
# the context of bayesian system's evaluation) based on experiments that have already been executed, just by utilizing the
# generated topic values.
# -- job needs to be done for coding optimal performance --


# ROS topics values are inserted as text files (data3 text files)
indic = np.loadtxt('/home/dimiubuntu/data_collection/data4/path/path1_20.text', delimiter=',', usecols=(0), unpack=True)
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
# Dis1 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis1_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
# Dis2 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis2_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
# Dis3 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis3_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
# Dis4 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis4_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson
# Dis5 = np.loadtxt('/home/dimiubuntu/data_collection/data4/euclidean/dis5_20.text', delimiter=',', usecols=(1), unpack=True)  # for RBII & Carlson

iter = len(Angle1)
print("iterations=" +str(iter))


robot_coord = [0, 0]  # robot's coordinates w.r.t robot's frame
maxA = 180
maxP = 25
n = 5   # number of goals
Delta = 0.2  # constant that defines the probabilities in transition/conditional table
wA = 0.6  # angle weight
wP = 0.4 # path weight
flag = 0
time = 10
P0 = 0.95  # goal's probability when click is active
threshold = 0.35
slope = (P0-threshold) / time
rest = (1-P0)/(n-1)  # probability of the remaining goals (these that are not being clicked)


CLICKS = np.loadtxt('/home/dimiubuntu/data_collection/data4/click/click_20.text', delimiter=',', usecols=(0), unpack=True)
print("number of clicks : " +str(len([i for i in indic if i in CLICKS])))  # how many clicks
print("ROS click time : " +str([i for i in indic if i in CLICKS]))  # confirm that clicks (ROS time values) exist based on any of the extracted text files

# find the positions where the clicks happened
Clicks = np.array([])
for k in CLICKS:
    CLICKS = [i for i, value in enumerate(indic, start=0) if value == k]
    Clicks = np.append(Clicks, CLICKS)  # float array
click = Clicks.astype(np.int64)  # convert to int array because we need it as index parameter
print("The clicks are located in positions : " +str(click))

# creation of Conditional Probability Table 'nxn' according to goals & Delta
data1 = np.ones((n, n)) * (Delta / (n-1))
np.fill_diagonal(data1, 1-Delta)
cond = data1
print("C.P.T. is : " +str(cond))


def compute_like(Angle, Path):
    a = Angle / maxA
    # print(a)
    p = Path / maxP
    # print(p)
    like = np.exp(-a/wA) * np.exp(-p/wP)
    return like


def compute_decay(n, j):
    # print('timing', j)
    decay = P0 - (slope * j)
    # updated = np.ones(n-1) * (1-decay)/(n-1)
    rest = (1 - decay) / (n - 1)
    if i < time and i >= click[0] and flag == 1:
        #updated = np.array([decay, rest, rest, rest, rest])
        #updated = np.array([rest, decay, rest, rest, rest])
        #updated = np.array([rest, rest, decay, rest, rest])
        #updated = np.array([rest, rest, rest, decay, rest])
        updated = np.array([rest, rest, rest, rest, decay])
    else:
        updated = np.array([decay, rest, rest, rest, rest])
        #updated = np.array([rest, decay, rest, rest, rest])
        #updated = np.array([rest, rest, decay, rest, rest])
        #updated = np.array([rest, rest, rest, decay, rest])
        #updated = np.array([rest, rest, rest, rest, decay])
    return updated

# compute conditional : normalized [SP(goal.t|goal.t-1) * b(goal.t-1)]
def compute_conditional(cond, prior):
    sum = np.matmul(cond, prior.T)
    return sum

# compute posterior P(goal|Z) = normalized(likelihood * conditional)
def compute_post(likelihood, summary):
    out2 = likelihood * summary
    post = out2 / np.sum(out2)
    return post

def extra_term(summary, dec):
    extra = summary * dec
    return extra

def compute_final(likelihood, plus):
    out = likelihood * plus
    poster = out / np.sum(out)
    return poster


# our MAIN loop
total = np.array([])  # array for saving the max indices (~which goal is most probable)
post_max = np.array([])  # array for saving the percentage value corresponding to max index
posterior_1 = np.array([])
posterior_2 = np.array([])
posterior_3 = np.array([])
posterior_4 = np.array([])
posterior_5 = np.array([])
prior = []
for i in range(len(indic)):
    Angle = np.array([Angle1[i], Angle2[i], Angle3[i], Angle4[i], Angle5[i]])
    Path = np.array([Path1[i], Path2[i], Path3[i], Path4[i], Path5[i]])
    if (i < time) and (i == click[0]) and (flag == 0):  # set prior when first click is active
        #prior = np.array([P0, rest, rest, rest, rest])  # if Initial goal=1 otherwise comment
        #prior = np.array([rest, P0, rest, rest, rest])  # if Initial goal=2 otherwise comment
        #prior = np.array([rest, rest, P0, rest, rest])  # if Initial goal=3 otherwise comment
        #prior = np.array([rest, rest, rest, P0, rest])  # if Initial goal=4 otherwise comment
        prior = np.array([rest, rest, rest, rest, P0])  # if Initial goal=5 otherwise comment
        flag = 1
    elif (i >= click[1]) and (i < click[1]+time) and (flag == 1):  # set prior when second click is active
        prior = np.array([P0, rest, rest, rest, rest])  # if Final goal=1 otherwise comment
        #prior = np.array([rest, P0, rest, rest, rest])  # if Final goal=2 otherwise comment
        #prior = np.array([rest, rest, P0, rest, rest])  # if Final goal=3 otherwise comment
        #prior = np.array([rest, rest, rest, P0, rest])  # if Final goal=4 otherwise comment
        #prior = np.array([rest, rest, rest, rest, P0])  # if Final goal=5 otherwise comment
        flag = 2


    if i < time and flag == 1 and i != click[1]:
        # print('mpes')
        likelihood = compute_like(Angle, Path)
        summary = compute_conditional(cond, prior)
        dec = compute_decay(n, i)
        plus = extra_term(summary, dec)
        posterior = compute_final(likelihood, plus)
        value_max = np.amax(posterior)
        post_max = np.append(post_max, value_max)
        index = np.argmax(posterior)
        print(
            "decay", "iter=" + str(i), "MAX:" + str(index + 1), "prior:" + str(prior), "posterior:" + str(posterior),
            "decay:" + str(dec))
        total = np.append(total, index + 1)
        prior = posterior

    elif (i >= click[1]) and (i < click[1] + time):
        # print('--------------')
        likelihood = compute_like(Angle, Path)
        summary = compute_conditional(cond, prior)
        dec = compute_decay(n, i - click[1])
        plus = extra_term(summary, dec)
        posterior = compute_final(likelihood, plus)
        value_max = np.amax(posterior)
        post_max = np.append(post_max, value_max)
        index = np.argmax(posterior)
        print(
            "decay", "iter=" + str(i), "MAX:" + str(index + 1), "prior:" + str(prior), "posterior:" + str(posterior),
            "decay:" + str(dec))
        total = np.append(total, index + 1)
        prior = posterior

    else:
        # print(i)
        likelihood = compute_like(Angle, Path)
        summary = compute_conditional(cond, prior)
        posterior = compute_post(likelihood, summary)
        value_max = np.amax(posterior)
        post_max = np.append(post_max, value_max)
        index = np.argmax(posterior)
        print(
            "normal", "iter=" + str(i), "MAX:" + str(index + 1), "prior:" + str(prior), "posterior:" + str(posterior))
        total = np.append(total, index + 1)
        prior = posterior

    posterior_1 = np.append(posterior_1, posterior[0])
    posterior_2 = np.append(posterior_2, posterior[1])
    posterior_3 = np.append(posterior_3, posterior[2])
    posterior_4 = np.append(posterior_4, posterior[3])
    posterior_5 = np.append(posterior_5, posterior[4])

print(total)


# algorithm's EVALUATION
# after bayes results make evaluation properly .. find the "corrects" on each group
def evaluation(iter, total):
    # run through total array and find correct on each group
    # values until first change --> find correct1 (here actual goal = 2)
    total1 = np.array([])
    for i in range(click[1]):  # new[0]-1
        total1 = np.append(total1, total[i])
    print("total1 : " + str(total1))
    correct1 = np.count_nonzero(total1 == 5)  # 4 .. 2 .. 2 .. 5 .. 2 .. 1 .. 1 .. 2 .. 2 .. 5 .. 4 .. 2 .. 1 .. 1 .. 1
    print("No of total1 measurements : " + str(len(total1)), "correct1 : " + str(correct1))
    print("------------------------------------------------------------------------------------------------------")

    # values from first "change" until end --> find correct2 (here actual goal = 1)
    total2 = np.array([])
    for i in range(click[1], iter):  # new[0]-1, iter
        total2 = np.append(total2, total[i])
    print("total2 : " + str(total2))
    correct2 = np.count_nonzero(total2 == 1)  # 3 .. 5 .. 3 .. 4 .. 4 .. 5 .. 2 .. 5 .. 3 .. 3 .. 1 .. 5 .. 5 .. 3 .. 2
    print("No of total2 measurements : " + str(len(total2)), "correct2 : " + str(correct2))
    print("------------------------------------------------------------------------------------------------------")

    Correct = correct1 + correct2  # sum of corrects
    success = ("{0:.2%}".format(Correct / iter))  # find the success rate (Correct/iterations)
    print("% of success: " + str(success))

    return success, click, total1, total2


# evaluate the "jitter" metric .. once the algorithm has captured the correct goal after change check if it will lose it till the end
def jitter_evaluation(total1, total2):
    value_correct1 = 5
    ind1 = np.where(total1 == value_correct1)[0]
    first_occurrence1 = ind1[0]
    print("first occurrence of correct=1 after change in", first_occurrence1)
    fail1 = len([i for i in total1[first_occurrence1:] if i != value_correct1])
    jitter1 = fail1 / (len(total1) - first_occurrence1)
    jitter1_perce = ("{0:.2%}".format(jitter1))
    print("how many fails1=",fail1, "," "jitter1=",jitter1_perce)

    value_correct2 = 1
    ind2 = np.where(total2 == value_correct2)[0]
    first_occurrence2 = ind2[0]
    print("first occurrence of correct=3 after change in", first_occurrence2)
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
        if i < click[1]:
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


