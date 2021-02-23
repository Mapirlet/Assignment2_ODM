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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from keras.models import Sequential
from keras.layers import Dense
from section1 import createInstanceDomain
from section2 import compute_expected_return


def Fitted_Q_iteration(domain,set_tuples,model,criteria):

    N = 0
    Q_prev = 0
    Q_prev2 = 0
    
    while criteria(N,domain.discount_factor,Q_prev,Q_prev2,set_tuples) is True:
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
        Q_prev2 = Q_prev
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

def stopping_criterion1(N,discount_factor,Q_N,Q_prev,set_tuples):
    Br = 1
    denom = (1-discount_factor)**2
    bound = (2*Br*discount_factor**N)/denom
    if bound < 1e-02:
        return False
    return True

def stopping_criterion2(N,discount_factor,Q_N,Q_prev,set_tuples):
    if (N == 0) or (N == 1):
        return True
    tolerance = 1
    summ = 0
    for j in range(len(set_tuples)):
        x_t = set_tuples[j][0]
        u_t = set_tuples[j][1]
        tmp1 = Q_N.predict([[x_t[0],x_t[1],u_t]])
        tmp2 = Q_prev.predict([[x_t[0],x_t[1],u_t]])
        summ = summ + abs(tmp1[0]-tmp2[0])
    if summ < tolerance:
        return False
    return True

def Linear_Regression(i,o):
    model = LinearRegression()
    model.fit(i,o)
    return model

def Extremely_Randomized_Trees(i,o):
    model = ExtraTreesRegressor(random_state=0)
    model.fit(i,o)
    return model

def Neural_Networks(i,o):
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(i,o)
    return model

def plot_all_from_Q(domain,Q_fct,resolution):
    
    p_vector = np.linspace(-1,1,int(2/resolution)+1)
    l_p = len(p_vector)
    s_vector = np.linspace(-3,3,int(6/resolution)+1)
    l_s = len(s_vector)
    actions = domain.actions
    l_a = len(actions)
    X, Y = np.meshgrid(p_vector,s_vector)
    Q_map = np.zeros([l_s,l_p,l_a])

    for i in range(l_p):
        for j in range(l_s):
            for k in range(l_a):
                p = p_vector[i]
                s = s_vector[j]
                u = actions[k]
                Q_map[j,i,k] = Q_fct.predict([[p,s,u]])
    
    color_map1 = Q_map[:,:,0]
    color_map2 = Q_map[:,:,1]
    min_v = np.min(color_map1)
    max_v = np.max(color_map1)
    fig1, ax1 = plt.subplots()
    c = ax1.contourf(X,Y,color_map1,cmap='RdBu',vmax=max_v,vmin=min_v)
    fig1.colorbar(c)
    ax1.set_title('U = 4')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Speed')
    
    min_v = np.min(color_map2)
    max_v = np.max(color_map2)
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(X,Y,color_map2,cmap='RdBu',vmax=max_v,vmin=min_v)
    fig2.colorbar(c)
    ax2.set_title('U = -4')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Speed')

    policy = np.zeros([l_s,l_p])

    for p in range(l_p):
        for s in range(l_s):
            index = np.argmax(Q_map[s,p])
            policy[s,p] = actions[index]
    
    min_v = -4
    max_v = 4
    fig3, ax3 = plt.subplots()
    c = ax3.contourf(X,Y,policy,cmap='RdBu',vmax=max_v,vmin=min_v)
    fig3.colorbar(c)
    ax3.set_title('Policy (blue = 4) / (red = -4)')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Speed')
    
    policy_object = MyPolicy(policy,resolution)
    return policy_object

class MyPolicy():
    def __init__(self,policy_grid,resolution):
        self.policy = policy_grid
        self.resolution = resolution
        
    def getPolicy(self,x):
        p_index = round((x[0]+1)/self.resolution)
        s_index = round((x[1]+3)/self.resolution)
        return self.policy[s_index,p_index]

if __name__ == "__main__":
   
   domain = createInstanceDomain(0.001) 
    
   F1 = create_set_tuples1(domain,100)
   F2 = create_set_tuples2(domain,100)
   
   #Q_LR = Fitted_Q_iteration(domain,F2,Linear_Regression,stopping_criterion1)
   Q_ERT = Fitted_Q_iteration(domain,F2,Extremely_Randomized_Trees,stopping_criterion1)
   #Q_NN = Fitted_Q_iteration(domain,F2,Neural_Networks,stopping_criterion1)
   
   #P_LR = plot_all_from_Q(domain,Q_LR,0.01)
   P_ERT = plot_all_from_Q(domain,Q_ERT,0.01)
   #P_NN = plot_all_from_Q(domain,Q_NN,0.01)
   
   # J_LR = compute_expected_return(domain,500,P_LR.getPolicy)
   # print(J_LR)
   J_ERT = compute_expected_return(domain,500,P_ERT.getPolicy)
   print(J_ERT)
   #J_NN = compute_expected_return(domain,500,P_NN.getPolicy)
   #print(J_NN)