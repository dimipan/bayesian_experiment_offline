from __future__ import division
import numpy as np

"""
Copyright (c) 2020-2021, Dimitris Panagopoulos
All rights reserved.
----------------------------------------------------
The proposed framework for operator intent recognition (BOIR) based on recursive Bayesian estimation as it is being
described in the paper "A Bayesian-based approach to Human-Operator Intent Recognition in Remote Mobile Robot
Navigation" published in the SMC 2021 
"""

class BayesianOperatorIntentRecognition:
    def __init__(self, n, Delta, Angle, wA, Path, wP, maxA, maxP, Dis, iteration):
        self.n = n
        self.Delta = Delta
        self.Angle = Angle
        self.wA = wA
        self.Path = Path
        self.wP = wP
        self.maxA = maxA
        self.maxP = maxP
        self.Dis = Dis
        self.iteration = iteration
        self.a = self.Angle / self.maxA
        self.p = self.Path / self.maxP
        self.cpt = np.ones((self.n, self.n)) * (self.Delta / (self.n - 1))
        np.fill_diagonal(self.cpt, 1 - self.Delta)

    def initialization_prior(self):
        if self.iteration == 0:
            prior = np.ones(self.n) * (1 / self.n)
        else:
            prior = post
        return prior

    def compute_like(self):
        like = np.exp(-self.a / self.wA) * np.exp(-self.p / self.wP)
        return like

    def compute_conditional(self, prior):
        summary = np.matmul(self.cpt, prior.T)
        return summary

    def compute_post(self, like, summary):
        result = like * summary
        post = result / np.sum(result)
        return post

    def get_maximium_value(self, post):
        max_index = np.argmax(post)
        return max_index




