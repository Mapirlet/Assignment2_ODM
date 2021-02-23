"""
University of Liege
INFO8003-1 - Optimal decision making for complex problems
Assignment 1 - Reinforcement Learning in a Discrete Domain
By:
    PIRLET Matthias
    CHRISTIAENS Nicolas
"""

import numpy as np
import matplotlib.pyplot as plt
from section1 import createInstanceDomain,policyRight

def compute_expected_return(domain, N, policy, p_init, s_init):
    """
    This function returns the expected return of a policy.
    Inputs: 
        - domain : Instance of Domain class
        - N : number of steps >= 1
        - policy : the policy function used
    Output: 
        The expected return of a policy
    """
    pos = p_init
    speed = s_init
    expected_return = 0

    for i in range(N):
        if domain.terminalState(pos, speed):
            break
        action = policy(pos, speed)
        if isinstance(action,str):
            action = domain.getAction(action)

        r = domain.getReward(pos, speed, action)
        expected_return += ((domain.discount_factor)**i)*r
        pos, speed = domain.getNextState(pos, speed, action)


    return expected_return

def generatePlot(domain, policy, m, max_n):

    y = []
    x = list(range(1, max_n+1))
    for i in range(1,max_n+1): #size of N
        J = []
        for j in range(1,m+1): #50
            p_0 = np.random.uniform(-0.1, 0.1)
            s_0 = 0
            expected_return = compute_expected_return(domain, i, policy, p_0, s_0)
            J.append(expected_return)
        value = sum(J)/len(J)
        y.append(value)

    plt.plot(x, y)
    plt.xlabel('Size of the trajectory')
    plt.ylabel('expected return')
    plt.show()

def generatePlot2(domain, policy, m, max_n):
    reward_matrix = np.zeros([m, max_n])
    x = list(range(1, max_n+1))
    for i in range(1,m+1): #50
        p_0 = np.random.uniform(-0.1, 0.1)
        s_0 = 0
        for j in range(1,max_n+1): #N
            reward_matrix[i-1,j-1] = compute_expected_return(domain, j, policy, p_0, s_0)

    rew = np.sum(reward_matrix, axis=0)
    y = rew/m
    plt.plot(x, y)
    plt.xlabel('Size of the trajectory')
    plt.ylabel('expected return')
    plt.show()

if __name__ == "__main__":
    domain = createInstanceDomain(0.001)
    generatePlot2(domain, policyRight, 50, 1000)