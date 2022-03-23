from __future__ import division
import numpy as np

"""
Copyright (c) 2020-2021, Dimitris Panagopoulos
All rights reserved.
----------------------------------------------------
The proposed framework for operator intent recognition (BOIR-AIRM) based on recursive Bayesian estimation as it is being
described in the paper "A Bayesian-based approach to Human-Operator Intent Recognition in Remote Mobile Robot
Navigation" published in the SMC 2021. 
"""

class BayesianOperatorIntentRecognition_AIRM:
    def __init__(self, n, Delta, Angle, wA, Path, wP, maxA, maxP, P0, threshold, time, iteration):
        self.n = n          # number of goals
        self.Delta = Delta  # constant that defines the probabilities in transition/conditional table
        self.Angle = Angle
        self.wA = wA        # angle weight
        self.Path = Path
        self.wP = wP        # path weight
        self.maxA = maxA    # max value that angle can take
        self.maxP = maxP    # max value that path can take
        self.P0 = P0        # goal's probability when click is active
        self.threshold = threshold  # minimum value that decay could take
        self.time = time
        self.iteration = iteration
        self.a = self.Angle / self.maxA
        self.p = self.Path / self.maxP
        self.cpt = np.ones((self.n, self.n)) * (self.Delta / (self.n - 1))
        np.fill_diagonal(self.cpt, 1 - self.Delta)
        self.slope = (self.P0-self.threshold) / self.time

    def compute_like(self):
        like = np.exp(-self.a / self.wA) * np.exp(-self.p / self.wP)
        return like

    def compute_decay_scenario_3(self, j, click):
        decay = self.P0 - (self.slope * j)
        rest = (1 - decay) / (self.n - 1)
        if self.iteration < self.time and self.iteration >= click[0]:
            updated = np.array([decay, rest, rest])
        elif (self.iteration >= click[1]) and (self.iteration < click[1] + self.time):
            updated = np.array([rest, decay, rest])
        else:
            updated = np.array([rest, rest, decay])
        return updated

    def compute_decay_scenario_4(self, j, click):
        decay = self.P0 - (self.slope * j)
        rest = (1 - decay) / (self.n - 1)
        if self.iteration < self.time and self.iteration >= click[0]: #and flag == 1:
            # updated = np.array([decay, rest, rest, rest, rest])
            # updated = np.array([rest, decay, rest, rest, rest])
            # updated = np.array([rest, rest, decay, rest, rest])
            # updated = np.array([rest, rest, rest, decay, rest])
            updated = np.array([rest, rest, rest, rest, decay])
        else:
            updated = np.array([decay, rest, rest, rest, rest])
            # updated = np.array([rest, decay, rest, rest, rest])
            # updated = np.array([rest, rest, decay, rest, rest])
            # updated = np.array([rest, rest, rest, decay, rest])
            # updated = np.array([rest, rest, rest, rest, decay])
        return updated

    def compute_conditional(self, prior):
        summary = np.matmul(self.cpt, prior.T)
        return summary

    def extra_term(self, summary, dec):
        extra = summary * dec
        return extra

    def compute_post(self, likelihood, summary):
        out2 = likelihood * summary
        post = out2 / np.sum(out2)
        return post

    def compute_final(self, likelihood, plus):
        result = likelihood * plus
        post = result / np.sum(result)
        return post

    def get_maximium_value(self, post):
        max_index = np.argmax(post)
        return max_index




