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
from section1 import createInstanceDomain,policyLeft

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
        action = policy((pos, speed))
        if isinstance(action,str):
            action = domain.getAction(action)
        new_pos, new_speed = domain.getNextState(pos, speed, action)
        r = domain.getReward(pos, speed, action, new_pos, new_speed)
        expected_return += ((domain.discount_factor)**i)*r
        pos = new_pos
        speed = new_speed
        
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
    plt.xlabel('N')
    plt.ylabel('Expected return')
    plt.show()

if __name__ == "__main__":

    domain = createInstanceDomain(0.001)

    generatePlot(domain, policyLeft, 50, 1000)