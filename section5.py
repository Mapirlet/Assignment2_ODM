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
from sklearn.neural_network import MLPRegressor
from section1 import createInstanceDomain
from section2 import compute_expected_return
from section4 import create_set_tuples2,MyPolicy


def Parametric_Q_learning(set_tuples,alpha):
    
    weight = 0
    l = len(set_tuples)
    
    for j in range(l):
        x_t = set_tuples[j][0]
        u_t = set_tuples[j][1]
        r_t = set_tuples[j][2]
        x_t_next = set_tuples[j][3]
        
        
        temp_diff = r_t 
    
    return 0

def Neural_Networks(i,o):
    model = MLPRegressor(hidden_layer_sizes=(9,9,9,9,9),solver = 'sgd',activation='tanh',random_state=0, max_iter=500)
    model.fit(i,o)
    return model

def derived_policy(domain,Q_fct,resolution,plot=False):
    
    p_vector = np.linspace(-1,1,int(2/resolution)+1)
    l_p = len(p_vector)
    s_vector = np.linspace(-3,3,int(6/resolution)+1)
    l_s = len(s_vector)
    actions = domain.actions
    l_a = len(actions)
    X, Y = np.meshgrid(p_vector,s_vector)
    policy = np.zeros([l_s,l_p])
    
    Q_map = np.zeros([l_s,l_p,l_a])
    for i in range(l_p):
        for j in range(l_s):
            for k in range(l_a):
                p = p_vector[i]
                s = s_vector[j]
                u = actions[k]
                Q_map[j,i,k] = Q_fct.predict(np.array([p,s,u]).reshape(-1,3))
    
    for p in range(l_p):
        for s in range(l_s):
            index = np.argmax(Q_map[s,p])
            policy[s,p] = actions[index]
      
    if plot == True:
        min_v = -4
        max_v = 4
        fig1, ax1 = plt.subplots()
        c = ax1.contourf(X,Y,policy,cmap='RdBu',vmax=max_v,vmin=min_v)
        fig1.colorbar(c)
        ax1.set_title('Policy (blue = 4) / (red = -4)')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Speed')

    policy_object = MyPolicy(policy,resolution)
    expected = compute_expected_return(domain,500,policy_object.getPolicy)
    
    return expected

def experimental_protocol():
    return 0

if __name__ == "__main__":
    
    domain = createInstanceDomain(0.001) 
    
    F2 = create_set_tuples2(domain,50)
    