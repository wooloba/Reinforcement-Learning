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
rows = 6
col = 9
wind = [0,0,0,1,1,1,2,2,1,0]
end = [7,3]
start = [0,3]
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
    global current_state,end
    '''
    8 actions, 0-UP, 1-DOWN, 2-LEFT, 3-RIGHT, 4-UL, 5-UR, 6-DL, 7-DR
    '''

    if action == 0:             #UP
        if current_state[1] < rows:
            current_state[1] += 1
    elif action ==1:            #DOWN
        if current_state[1] > 0:
            current_state[1] -= 1
    elif action == 2:           #LEFT
        if current_state[0] > 0:
            current_state[0] -= 1
    elif action == 3:           #RIGHT
        if current_state[0] < col:
            current_state[0] += 1
    elif action == 4:           #UP LEFT
        if current_state[1] < rows and current_state[0] > 0:
            current_state[0] -= 1
            current_state[1] += 1
    elif action == 5:           #UP RIGHT
        if current_state[1] < rows and current_state[0] < col:
            current_state[0] += 1
            current_state[1] += 1
    elif action == 6:           #DOWN LEFT
        if current_state[1] > 0 and current_state[0] > 0:
            current_state[0] -= 1
            current_state[1] -= 1
    elif action == 7:           #DOWN RIGHT
        if current_state[1] > 0 and current_state[0] < col:
            current_state[0] += 1
            current_state[1] -= 1
    elif action == 8:
        pass

    #wind power applied
    wind_power = wind[current_state[0]]
    current_state[1] += wind_power
    if current_state[1] >= rows:
        current_state[1] = rows

    if current_state == [7,3]:
        is_terminal = True
        reward = 0.0
    else:
        is_terminal = False
        reward = - 1.0

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
