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
    def __init__(self,domain,alpha,epsilon,Norm=False,K=10,N=100):
        self.domain = domain
        self.alpha = alpha
        self.epsilon = epsilon
        self.target = None
        self.eval = None
        self.buffer = []
        self.Norm = Norm
        self.K = K
        self.N = N

    def greedy_policy(self,x):
        """
        Inputs: 
            - x : a state
        Output: 
            The greedy policy
        """
        actions = self.domain.actions
        rand = np.random.uniform()
        if (rand > self.epsilon) and (len(self.buffer) != 0):
            index = np.argmax(self.eval.predict(np.asarray(x).reshape(-1,2)))
        else:
            index = np.random.choice(len(actions))
        return actions[index]

    def random_sample(self):
        if len(self.buffer) == 1:
            rand = 0
        else :
            rand = np.random.randint(0,len(self.buffer)-1)
        return self.buffer[rand]
    
    def Save_target_parameters(self):
        self.target.set_weights(self.eval.get_weights())
        
    def Init_network(self):
        self.target = Neural_Networks(2,len(self.domain.actions),self.alpha)
        self.eval = Neural_Networks(2,len(self.domain.actions),self.alpha)
    
    def CollectData(self):
        p= np.random.uniform(-1, 1)
        s = 0
        steps_max = 100
        for i in range(steps_max):
            if domain.terminalState(p,s):
                break
            action = self.greedy_policy((p,s))
            sample = self.domain.generateTrajectory((p,s),action,i)
            (p,s) = sample[3]
            self.buffer.append(sample)
    
    def Create_train(self,sample):
        x_t = sample[0]
        u_t = sample[1]
        r_t = sample[2]
        x_t_next = sample[3]
        
        Q_eval = self.eval.predict(np.asarray(x_t_next).reshape(-1,2))[0]
        max_action = np.argmax(Q_eval)
        if domain.terminalState(x_t_next[0],x_t_next[1]):
            Q_next = [0,0]
        else:
            Q_next = self.target.predict(np.asarray(x_t_next).reshape(-1,2))[0]   
        index = self.domain.actions.index(u_t)
        Q_target = self.eval.predict(np.asarray(x_t).reshape(-1,2))[0]
        Q_target[index] = r_t + self.domain.discount_factor*Q_next[max_action]
        
        return np.asarray(Q_target)
        
    def Core_Simulation(self):
        self.Init_network()
        for i in range(self.N):
            self.CollectData()
            X_train = []
            Y_train = []
            for j in range(self.K):
                sample = self.random_sample()
                X_train.append(np.asarray(sample[0]))
                Q_target = self.Create_train(sample)
                Y_train.append(Q_target)
            X_train = np.asarray(X_train)
            Y_train = np.asarray(Y_train)
            
            if self.Norm is False:
                self.eval.fit(X_train,Y_train)
            else:
                self.NormLearn()
            self.Save_target_parameters()
    
    def NormLearn(self):
        """
        Not Done
        """
        return 0
    
def Neural_Networks(input_dim,output_dim,alpha,Nb_neurons = 100):
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
    

if __name__ == "__main__":
    
    domain = createInstanceDomain(0.001) 
    
    F = create_set_tuples2(domain,50)
    
    Not_norm = Online_Q_iteration(domain,0.05,0.25)
    Not_norm.Core_Simulation()