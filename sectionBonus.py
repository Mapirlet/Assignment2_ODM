"""
University of Liege
INFO8003-1 - Optimal decision making for complex problems
Assignment 2 - Reinforcement Learning in a Continuous Domain
By:
    PIRLET Matthias
    CHRISTIAENS Nicolas
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf
from section1 import createInstanceDomain
from section2 import compute_expected_return
from section4 import create_set_tuples2,MyPolicy


class Online_Q_iteration():
    def __init__(self,domain,alpha,epsilon,Norm=False):
        self.domain = domain
        self.alpha = alpha
        self.epsilon = epsilon
        self.NN_target = None
        self.NN_eval = None
        self.buffer = []
        self.Norm = Norm

    def greedy_policy(self,x):
        """
        Inputs: 
            - x : a state
        Output: 
            The greedy policy
        """
        actions = self.domain.actions
        rand = np.random.uniform()
        if (rand > self.epsillon):
            index = np.argmax(self.Q_eval.predict(x))
        else:
            index = np.random.choice(len(actions))
        return actions[index]

if __name__ == "__main__":
    
    domain = createInstanceDomain(0.001) 
    
    F = create_set_tuples2(domain,50)
    