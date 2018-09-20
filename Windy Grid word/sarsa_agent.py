#!/usr/bin/env python

"""
  Author: Yaozhi Lu

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

    #initialize the policy array in a smart way
    global Q,agent_last_state,agent_last_action
    Q = np.zeros((board_col,board_row,num_action))
    agent_last_state = [0,3]
    agent_last_action = 0
    return

def agent_start(state):
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
    # do learning and update pi
    global Q,agent_last_action,agent_last_state

    xx = agent_last_state[0]
    yy = agent_last_state[1]
    Q_SA = Q[xx][yy][agent_last_action]
    Q[xx][yy][agent_last_action] = Q_SA + alpha * (reward - Q_SA)
    return

def agent_cleanup():
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return
    else:
        return "I don't know what to return!!"

