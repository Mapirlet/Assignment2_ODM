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


def Fitted_Q_iteration(domain,set_tuples,model,criteria):

    N = 0
    Q_prev = 0
    
    while criteria() is True:
        N = N + 1
        i = []
        o = []
        for i in range(len(set_tuples)):
            x_t = set_tuples[i][0]
            u_t = set_tuples[i][1]
            r_t = set_tuples[i][2]
            x_t_next = set_tuples[i][3]
            if N == 1:
                i.append((x_t,u_t))
                o.append(r_t)
            else:
                Q_max = -np.inf
                for u in domain.actions:
                    Q_value = Q_prev.predict(x_t_next,u)
                    if Q_value > Q_max:
                        Q_max = Q_value
                i.append((x_t,u_t))
                o.append(r_t+domain.discount_factor*Q_max)
            Q_prev = model(o,i)
    
    return Q_prev

def random_Policy(x):
    rand = np.random.rand()
    if rand < 0.5:
        return -4
    else:
        return 4

def create_set_tuples1(domain,N):
    
    set_tuples = []
    
    for i in range(N):
        p = np.random.uniform(-0.1, 0.1)
        s = 0
        steps_max = 100
        for i in range(steps_max):
            if domain.terminalState(p,s):
                break
            action = random_Policy()
            sample = domain.generateTrajectory((p,s),action,i)
            (p,s) = sample[3]
            set_tuples.append(sample)
                
    return set_tuples

def create_set_tuples2(domain,N):
    
    set_tuples = []
    
    for i in range(N):
        p = np.random.uniform(-1, 1)
        s = 0
        steps_max = 100
        for i in range(steps_max):
            if domain.terminalState(p,s):
                break
            action = random_Policy()
            sample = domain.generateTrajectory((p,s),action,i)
            (p,s) = sample[3]
            set_tuples.append(sample)
    
    return set_tuples

def stopping_criteria1(N,discount_factor):
    Br = 1
    denom = (1-discount_factor)**2
    bound = (2*Br*discount_factor**N)/denom
    if bound < 1e-05:
        return False
    return True

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
   
   