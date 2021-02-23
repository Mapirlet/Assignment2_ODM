"""
University of Liege
INFO8003-1 - Optimal decision making for complex problems
Assignment 1 - Reinforcement Learning in a Discrete Domain
By:
    PIRLET Matthias
    CHRISTIAENS Nicolas
"""

import numpy as np
import imageio
from section1 import createInstanceDomain,policyRight,simulateTrajectory
from display_caronthehill import save_caronthehill_image

def generate_images(trajectory):
   
    len_traj = len(trajectory)
    images =[]
    
    for i in range(len_traj):
        p = trajectory[i][0]
        s = trajectory[i][1]
        file_name = 'image/section3_{}.jpg'.format(i)
        save_caronthehill_image(p, s , file_name)

        images.append(imageio.imread(file_name))

    imageio.mimsave('trajectory.gif', images)

if __name__ == '__main__':

    domain = createInstanceDomain(0.001)
    trajectory = simulateTrajectory(policyRight, domain, 10,1)

    generate_images(trajectory)



