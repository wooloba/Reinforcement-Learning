#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import random_policy,rand_in_range, rand_un
import numpy as np
import pickle
from tiles3 import IHT,tiles

alpha = 0.01/50
gama = 1.0
iht = IHT((1/0.2+1)*(1/0.2+1)*50)
def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    # input policy here
    global policy,w,V

    w = np.zeros(1000)
    V = np.zeros(1000)
    return



def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global agent_last_state,w

    #take action base on policy
    action = random_policy()
    agent_last_state = state

    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global w,V,agent_last_state

    #last affected states list
    last_state_tile = (agent_last_state-1)/200.0
    tilings_last = tiles(iht, 50, [last_state_tile])

    #this affected states list
    state_tile = (state-1)/200.0
    tilings = tiles(iht,50,[state_tile])


    vhat_S_prime = np.zeros(1000)
    vhat_S = np.zeros(1000)

    for tile in tilings:
        vhat_S_prime[tile] = 1
    for tile in tilings_last:
        vhat_S[tile] = 1

    deri_vhat = vhat_S

    vhat_S_prime = np.dot(w, vhat_S_prime)
    vhat_S = np.dot(w,vhat_S)

    w = w + alpha*(reward + gama*vhat_S_prime - vhat_S)*deri_vhat

    V[agent_last_state-1] = np.dot(w,deri_vhat)
    action = random_policy()
    agent_last_state = state

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global w,V,agent_last_state

    last_state_tile = (agent_last_state-1)/200.0
    tilings_last = tiles(iht,50,[last_state_tile])

    vhat_S = np.zeros(1000)
    for tile in tilings_last:
        vhat_S[tile] = 1

    deri_vhat = vhat_S
    vhat_S = np.dot(w, vhat_S)

    w = w + alpha * (reward  - vhat_S) * deri_vhat

    V[agent_last_state - 1] = np.dot(w, deri_vhat)

    fp = open("shared.pkl", "w")
    fp.truncate()
    pickle.dump(V, fp)
    fp.close()
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
        return
    else:
        return "I don't know what to return!!"

