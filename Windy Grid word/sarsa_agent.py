#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

epsilon = 0.1
alpha = 0.5

'''
    change here to include or exclude 9th action
    8: excluded, 9:included
'''
num_action = 9
board_row = 7
board_col = 10

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    #initialize the policy array in a smart way
    global Q,agent_last_state,agent_last_action
    Q = np.zeros((board_col,board_row,num_action))
    agent_last_state = [0,3]
    agent_last_action = 0
    return

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global agent_last_action,agent_last_state
    x = state[0]
    y = state[1]
    if rand_un()<epsilon:
        action = rand_in_range(num_action)
    else:
        action = np.argmax(Q[x][y])

    agent_last_state = [0,3]
    agent_last_action = action
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global Q,agent_last_action,agent_last_state
    x = state[0]
    y = state[1]

    xx = agent_last_state[0]
    yy = agent_last_state[1]

    # select an action, based on Q
    if rand_un()< epsilon :
        action = rand_in_range(num_action)
    else:
        #pick the largest q in Q
        action = np.argmax(Q[x][y])
    Q_SA = Q[xx][yy][agent_last_action]
    Q[xx][yy][agent_last_action] = Q_SA + alpha * (reward + Q[x][y][action] - Q_SA)

    agent_last_action = action
    agent_last_state = [x,y]

    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global Q,agent_last_action,agent_last_state

    xx = agent_last_state[0]
    yy = agent_last_state[1]
    Q_SA = Q[xx][yy][agent_last_action]
    Q[xx][yy][agent_last_action] = Q_SA + alpha * (reward - Q_SA)
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
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

