"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
"""

from utils import rand_in_range, rand_un
import numpy as np
from tiles3 import IHT,tiles

numTilings = 8
alpha = 0.1 / numTilings
gama = 1.0
iht = IHT(4096)
lam = 0.9
weight_size = 4096

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    # input policy here
    global policy,w

    #the size of weight may need to be modified
    w = np.random.uniform(-0.001,0.0,weight_size)

    return



def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global agent_last_state,agent_last_action,w,z

    #since epsilon is zero in this case, agent dose not explore


    action = chooseAction(state)
    z = np.zeros(weight_size)


    x =state[0]
    y = state[1]
    agent_last_state = [x,y]
    agent_last_action = action

    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global z,w,agent_last_state,agent_last_action
    x = state[0]
    y = state[1]

    delta = reward

    tileList = getTiles(agent_last_state[0],agent_last_state[1],agent_last_action)
    for index in tileList:
        delta -= w[index]
        z[index] = 1

    action = chooseAction(state)

    tileList_prime = getTiles(x,y,action)
    for index in tileList_prime:
        delta+=gama*w[index]


    w += alpha*delta*z
    z = gama*lam*z

    agent_last_state = [x,y]
    agent_last_action = action


    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global w,agent_last_state,agent_last_action,z

    delta = reward
    tileList = getTiles(agent_last_state[0], agent_last_state[1], agent_last_action)
    for index in tileList:
        delta -= w[index]
        z[index] = 1.0

    w += alpha*delta*z
    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up

    return


def agent_message(in_message):  # returns string, in_message: string
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        print(w)
        pos_max = 0.5
        pos_min = -1.2
        vel_max = 0.07
        vel_min = -0.07
        position = np.arange(pos_min, pos_max, (pos_max - pos_min) / 50.0)
        velocity = np.arange(vel_min, vel_max, (vel_max - vel_min) / 50.0)

        Z = np.zeros((50, 50))
        for i in range(len(position)):
            for j in range(len(velocity)):
                Z[i, j] = (costToGo(position[i], velocity[j]))

        np.save("costToGo", Z)

        return
    else:
        return "I don't know what to return!!"


def getTiles(position,velocity,action):
    #each tile covering 1/8th of the bounded distance in each dimension
    pos = 0.5 - (-1.2)
    vel = 0.07 - (-0.07)
    return tiles(iht, numTilings, [(numTilings/pos)*position,(numTilings/vel)*velocity], [action])

def getValues(position,velocity,action):
    return np.sum(w[getTiles(position,velocity,action)])

def chooseAction(state):
    value = []
    for i in range(0,3):
        value.append(getValues(state[0],state[1],i))
    action = np.argmax(value)

    return action

def costToGo(position,velocity):
    value = []
    for action in [0,1,2]:
        value.append(getValues(position, velocity, action))
    print(value)
    return -np.max(value)
