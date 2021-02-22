"""
University of Liege
INFO8003-1 - Optimal decision making for complex problems
Assignment 2 - Reinforcement Learning in a Continuous Domain
By:
    PIRLET Matthias
    CHRISTIAENS Nicolas
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from keras.models import Sequential
from keras.layers import Dense
from section1 import createInstanceDomain


def Fitted_Q_iteration(set_tuples,model,criteria):

    N = 0
    Q_N = 0
    
    while criteria() is True:
        N = N + 1
        TS = []
        for i in range(len(set_tuples)):
            x_t = set_tuples[0]
            u_t = set_tuples[1]
            r_t = set_tuples[2]
            x_t_next = set_tuples[3]
            i = 0
            o = 0
    
    return Q_N

def random_Policy(x):
    rand = np.random.rand()
    if rand < 0.5:
        return -4
    else:
        return 4

def create_set_tuples1(domain):
    return

def create_set_tuples2(domain):
    return

def stopping_criteria1():
    return

def stopping_criteria2():
    return

def Liner_Regression():
    return

def Extremely_Randomized_Trees():
    return

def Neural_networks():
    return

if __name__ == "__main__":
   
   domain = createInstanceDomain(0.001) 
   
   