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
    def __init__(self, m, g,dis_time,integ_ts, discount_factor):
    	"""
        This function initializes the Domain instance.
        Inputs: 
            - m : mass of the car 
            - g : gravitational constant
            - dis_time : discrete time 
            - integ_ts : integration time step
            - discount_factor :
        """   
    	self.m = m 
    	self.g = g
    	self.dis_time = dis_time
    	self.integ_ts = integ_ts
    	self.discount_factor = discount_factor
        self.actions = [-4,4]


	def hillFunction(self, p):
		"""
	    This function computes the value of the Hill function given in the assignment 2
	    Inputs: 
	        - p : position of the car on the hill
	    Output: 
	        Value of the Hill function
	    """
		if p < 0:
			value = p**2 + p
			return value
		else:
			value = p/sqrt(1+5*p**2)
			return value


	def hillFunctionDeriv(self,p):
		"""
	    This function computes the value of the 1st derivative of the Hill 
	    		function given in the assignment 2
	    Inputs: 
	        - p : position of the car on the hill
	    Output: 
	        Value of the Hill 1st derivative function
	    """
		if p < 0:
			value = 2*p + 1
			return value
		else: 
			value = 1/((1+5*(p**2))**(3/2))
			return value


	def hillFunctionDeriv2(self,p):
		"""
	    This function computes the value of the 2nd derivative of the Hill 
	    		function given in the assignment 2
	    Inputs: 
	        - p : position of the car on the hill
	    Output: 
	        Value of the Hill 2nd derivative function
	    """
		if p < 0:
			value = 2
			return value
		else:
			value = (-15*p)/((1+5*(p**2))**(3/2))
			return value
    

    def terminalState(self,p,s):
    	"""
	    This function returns True if the state is terminal and false otherwise
	    Inputs: 
	        - p : position of the car on the hill
	        - s : speed of the car on the hill 
	    Output: 
	        boolean that represents if we are or not in a terminal state
	    """
    	if abs(p) > 1 or abs(s) > 3: 
    		return True
    	else: 
    		return False

    def dynamicFunction(self,p,s,u):
    	"""
	    This function computes the speed and the acceleration of the car on the hill
	    		that represent the dynamics
	    Inputs: 
	        - p : position of the car on the hill
	        - s : speed of the car on the hill 
	        - u : the action taken 
	    Output: 
	        2 values representing the speed and the value of the acceleration 
	    """
    	speed = s

    	common_denom = 1 + (self.hillFunctionDeriv(p)**2)
    	term1_acceleration = u/(self.m*common_denom)
    	term2_acceleration = (self.g*self.hillFunctionDeriv(p))/common_denom
    	term3_acceleration = ((s**2)*self.hillFunctionDeriv(p)*self.hillFunctionDeriv2(p))/ \
    						common_denom
    	acceleration = term1_acceleration - term2_acceleration - term3_acceleration

    	return speed, acceleration


    def getNextState(self,p,s,u):
    	"""
	    This function computes the next state of the car given its state and an action 
	    Inputs: 
	        - p : position of the car on the hill
	        - s : speed of the car on the hill 
	        - u : the action taken 
	    Output: 
	        2 values representing the new state of the car 
	    """
    	nb_steps = self.dis_intval/self.self.integ_ts
    	new_p = p 
    	new_s = s

    	for i in range(nb_steps):
    		if self.terminalSate(new_p, new_s):
    			return new_p, new_s
    		else:
    			tmp = self.dynamicFunction(p,s,u)
    			new_p = self.integ_ts*tmp[0] + new_p
    			new_s = self.integ_ts*tmp[1] + new_s

    	return new_pos, new_speed


    def getReward(self, p, s, u):
    	"""
	    This function computes the reward that the car will get given its state and an action 
	    Inputs: 
	        - p : position of the car on the hill
	        - s : speed of the car on the hill 
	        - u : the action taken 
	    Output: 
	        Reward received by the car by taking this action  
	    """
    	new_p, new_s = self.getNextState(p,s,u)

    	if new_p < -1 or abs(new_s) > 3:
    		return -1
    	elif new_p > 1 and abs(new_s) <= 3:
    		return 1
    	else:
    		return 0


def createInstanceDomain(integ_ts):
    """
    This function creates the instance of the domain described in the assigment 2.
    Inputs: 
        - integ_ts : value of the integration time step
    Output: 
        The instance of the domain
    """
    
    m = 1
    g = 9.81
    dis_time = 0.1
    discount_factor = 0.95

    domain = Domain(m, g, dis_time, integ_ts, discount_factor)
    return domain


if __name__ == "__main__":
	createInstanceDomain(0.001)
