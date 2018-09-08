#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np


current_state = None
rows = 5
col = 8
end = [8,5]
start = [0,3]
blocks = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]]
def env_init():
    global current_state
    current_state = np.zeros(2)


def env_start():
    """ returns numpy array """
    global current_state,start
    x=start[0]
    y=start[1]
    current_state = [x,y]   # start at S
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state,end,blocks
    '''
    4 actions, 0-UP, 1-DOWN, 2-LEFT, 3-RIGHT
    '''
    next_state = [current_state[0], current_state[1]]

    if action == 0:             #UP
        next_state[1] += 1
        if next_state in blocks or next_state[1] > rows:
            next_state[1] -= 1

    elif action ==1:            #DOWN
        next_state[1] -= 1
        if next_state in blocks or next_state[1] < 0 :
            next_state[1] += 1

    elif action == 2:           #LEFT
        next_state[0] -= 1
        if next_state in blocks or next_state[0] < 0:
            next_state[0] += 1

    elif action == 3:           #RIGHT
        next_state[0] += 1
        if next_state in blocks or next_state[0] > col:
            next_state[0] -= 1

    current_state[0] = next_state[0]
    current_state[1] = next_state[1]

    if current_state == [8,5]:
        is_terminal = True
        reward = 1.0
    else:
        is_terminal = False
        reward = 0.0


    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
