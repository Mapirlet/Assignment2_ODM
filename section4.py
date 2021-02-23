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
    
    while criteria(N,domain.discount_factor) is True:
        N = N + 1
        l = len(set_tuples)
        i = []
        o = []
        for j in range(l):
            x_t = set_tuples[j][0]
            u_t = set_tuples[j][1]
            r_t = set_tuples[j][2]
            x_t_next = set_tuples[j][3]
            if N == 1:
                i.append([x_t[0],x_t[1],u_t])
                o.append(r_t)
            else:
                Q_max = -np.inf
                for u in domain.actions:
                    Q_value = Q_prev.predict([[x_t_next[0],x_t_next[1],u]])
                    Q_value = Q_value[0]
                    if Q_value > Q_max:
                        Q_max = Q_value
                i.append([x_t[0],x_t[1],u_t])
                o.append(r_t+domain.discount_factor*Q_max)
        Q_prev = model(i,o)
    
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
            action = random_Policy((p,s))
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
            action = random_Policy((p,s))
            sample = domain.generateTrajectory((p,s),action,i)
            (p,s) = sample[3]
            set_tuples.append(sample)
    
    return set_tuples

def stopping_criteria1(N,discount_factor):
    Br = 1
    denom = (1-discount_factor)**2
    bound = (2*Br*discount_factor**N)/denom
    if bound < 1e-02:
        return False
    return True

def stopping_criteria2():
    
    return True

def Linear_Regression(i,o):
    model = LinearRegression()
    model.fit(i,o)
    return model

def Extremely_Randomized_Trees():
    return

def Neural_networks():
    return

if __name__ == "__main__":
   
   domain = createInstanceDomain(0.001) 
    
   F1 = create_set_tuples1(domain,100)
   F2 = create_set_tuples2(domain,100)
   
   Q_LR = Fitted_Q_iteration(domain,F1,Linear_Regression,stopping_criteria1)
   Q_ERT = 0
   Q_NN = 0