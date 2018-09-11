#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import random_policy,rand_in_range, rand_un
import numpy as np
import pickle


alpha = 0.5
gama = 1.0

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    # input policy here
    global policy,w,xs
    w = np.zeros(1000)
    xs = np.identity(1000)
    return



def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global agent_last_state,w,xs


    #take action base on policy
    action = random_policy()
    agent_last_state = state
    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global w,xs,agent_last_state
    vhat_raw = xs[state-1]
    last_vhat_raw = xs[agent_last_state-1]
    deri_vhat = xs[agent_last_state-1]


    vhat_raw = np.dot(w, vhat_raw)
    vhat_raw *= gama

    last_vhat_raw = np.dot(w,last_vhat_raw)

    w = w + alpha*(reward + vhat_raw - last_vhat_raw)*deri_vhat

    action = random_policy()
    agent_last_state = state

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global w,xs,agent_last_state

    last_vhat_raw = xs[agent_last_state - 1]
    deri_vhat = xs[agent_last_state-1]

    last_vhat_raw = np.dot(w, last_vhat_raw)

    w = w + alpha * (reward  - last_vhat_raw) * deri_vhat

    fp = open("shared.pkl", "w")
    fp.truncate()
    pickle.dump(w, fp)
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

