"""
University of Liege
INFO8003-1 - Optimal decision making for complex problems
Assignment 1 - Reinforcement Learning in a Discrete Domain
By:
    PIRLET Matthias
    CHRISTIAENS Nicolas
"""

import numpy as np

class Domain:
    def __init__(self, m, g,dis_intval,integ_ts, discount_factor):

    	self.m = m 
    	self.g = g
    	self.dis_intval = dis_intval
    	self.integ_ts = integ_ts
    	self.discount_factor = discount_factor
        self.actions = [-4,4]

    def HillFunction(self, p):
    	if p<0:
    		value = p**2 + p
    		return value
    	else:
    		value = p/sqrt(1+5*p**2)
    		return value

    def HillFunction1stDeriv(self,p):
    	
    

    def getNextState(self,p,s,u):



    	return new_p, new_s


    def getReward(self, p, s, u):
    	new_p, new_s = self.getNextState(p,s,u)



def createInstanceDomain(integ_ts):
    """
    This function creates the instance of the domain described in the assigment 2.
    Inputs: 
        - integ_ts : value of the integration time stamp
    Output: 
        The instance of the domain
    """
    
    m = 1
    g = 9.81
    dis_intval = 0.1
    discount_factor = 0.95

    domain = Domain(m, g, dis_intval, integ_ts, discount_factor)
    return domain


if __name__ == "__main__":
	createInstanceDomain(0.001)
