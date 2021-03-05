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
from section1 import createInstanceDomain
from section2 import compute_expected_return
from section4 import create_set_tuples2,MyPolicy,FQI


def Parametric_Q_learning(domain,set_tuples,alpha):
    """
    Inputs: 
        - domain : domain instance (domain of the problem)
        - set_tuples : set of tuple of form (x_t,u_t,r_t,x_t+1)
        - alpha : learning rate
    Output: 
        The Neural Network object obtained with the parametric Q learning algorithm
    """
    
    NN = Neural_Networks(2,2,alpha)
    l = len(set_tuples)
    actions = domain.actions
    
    for j in range(l):
        x_t = set_tuples[j][0]
        u_t = set_tuples[j][1]
        r_t = set_tuples[j][2]
        x_t_next = set_tuples[j][3]
        
        Q_eval = NN.predict(np.asarray(x_t).reshape(-1,2))[0]
        
        if domain.terminalState(x_t_next[0],x_t_next[1]):
            Q_next = 0
        else:
            Q_next = NN.predict(np.asarray(x_t_next).reshape(-1,2))[0]
        
        index = actions.index(u_t)
        Q_target = Q_eval.copy()
        Q_target[index] = r_t + domain.discount_factor*np.max(Q_next)
        
        NN.fit(np.asarray(x_t).reshape(-1,2),np.asarray(Q_target).reshape(-1,2))
        
    return NN

def Neural_Networks(input_dim,output_dim,alpha,Nb_neurons = 10):
    """
    Inputs: 
        - input_dim : dimension of the inputs
        - output_dim : dimension of the outputs
        - alpha : learning rate
        - NB_neurons : number of neurons on the hidden layers
    Output: 
        A neural network
    """
    model = Sequential([
                Dense(Nb_neurons,input_shape=(input_dim,)),
                Activation('relu'),
                Dense(Nb_neurons),
                Activation('relu'),
                Dense(output_dim)])
    opt = SGD(learning_rate=alpha)
    model.compile(optimizer=opt,loss='mse')
    return model

def derived_policy(domain,Q_fct,resolution,plot=False):
    """
    Inputs: 
        - domain : domain instance (domain of the problem)
        - Q_fct : Q function approximator
        - resolution : resolution requiered for precision
        - plot : if True plot the grid of policy according to resolution
    Output: 
        The policy derived of Q_fct
    """
    p_vector = np.linspace(-1,1,int(2/resolution)+1)
    l_p = len(p_vector)
    s_vector = np.linspace(-3,3,int(6/resolution)+1)
    l_s = len(s_vector)
    actions = domain.actions
    l_a = len(actions)
    X, Y = np.meshgrid(p_vector,s_vector)
    policy = np.zeros([l_s,l_p])
    
    Q_map = np.zeros([l_s,l_p,l_a])
    Q_tmp = []
    index = 0
    for i in range(l_p):
        for j in range(l_s):
            p = p_vector[i]
            s = s_vector[j]
            Q_tmp.append(np.asarray([p,s]))
            index = index + 1
    Q_tmp = np.asarray(Q_tmp)
    Q_tmp =  Q_fct.predict(Q_tmp)
    for i in range(l_p):
        for j in range(l_s):
            for k in range(l_a):
                Q_map[j,i,k] = Q_tmp[i*l_s+j][k]
    
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
    
    return policy_object

def experimental_protocol(domain,resolution):
    """
    Experimental protocol of the section 5
    """
    X = [1,5,10,15,20]
    Y_FQI = np.zeros(len(X))
    Y_Param = np.zeros(len(X))
    
    for i in range(len(X)):
        nb_transtions = X[i]
        F = create_set_tuples2(domain,nb_transtions)
        
        FQI_ = FQI(domain,F)
        FQI_policy = GetFQI_policy(domain, FQI, resolution)
        Y_FQI[i] = compute_expected_return(domain,500,FQI_policy.getPolicy)
        
        Param = Parametric_Q_learning(domain,F,0.05)
        Param_policy = derived_policy(domain,Param,resolution,plot=False)
        Y_Param[i] = compute_expected_return(domain,500,Param_policy.getPolicy)
    
    return 0

def GetFQI_policy(domain,FQI,resolution):    
    """
    Inputs: 
        - domain : domain instance (domain of the problem)
        - FQI : Fitted Q Iteration approximator (section 4)
        - resolution : resolution requiered for precision
    Output: 
        The policy derived of Q_fct
    """
    p_vector = np.linspace(-1,1,int(2/resolution)+1)
    l_p = len(p_vector)
    s_vector = np.linspace(-3,3,int(6/resolution)+1)
    l_s = len(s_vector)
    actions = domain.actions
    l_a = len(actions)
    X, Y = np.meshgrid(p_vector,s_vector)
    policy = np.zeros([l_s,l_p])
    
    Q_map = np.zeros([l_s,l_p,l_a])
    Q_tmp = []
    index = 0
    for i in range(l_p):
        for j in range(l_s):
            for k in range(l_a):
                p = p_vector[i]
                s = s_vector[j]
                u = actions[k]
                Q_tmp.append(np.asarray([p,s,u]))
                index = index + 1
    Q_tmp = np.asarray(Q_tmp)
    Q_tmp =  FQI.predict(Q_tmp)
    for i in range(l_p):
        for j in range(l_s):
            for k in range(l_a):
                Q_map[j,i,k] = Q_tmp[i*l_s+j*l_a+k]
    
    for p in range(l_p):
        for s in range(l_s):
            index = np.argmax(Q_map[s,p])
            policy[s,p] = actions[index]
    
    policy_object = MyPolicy(policy,resolution)
    
    return policy_object

if __name__ == "__main__":
    
    domain = createInstanceDomain(0.001) 
    
    # F = create_set_tuples2(domain,100)
    # NN = Parametric_Q_learning(domain,F,0.2)
    # Param_policy = derived_policy(domain,NN,0.01,plot=True)
    # expected = compute_expected_return(domain,500,Param_policy.getPolicy)
    # print(expected)
    
    experimental_protocol(domain,0.01)