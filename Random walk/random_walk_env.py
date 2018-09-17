#!/usr/bin/env python

"""
  Author: Yaozhi Lu
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np


current_state = None
start = 500

def env_init():
    global current_state
    current_state = 0


def env_start():
    """ returns numpy array """
    global current_state,start
    current_state = start   # start at S
    return current_state

def env_step(action):
    global current_state,end

    current_state += action

    if current_state >= 1001:
        current_state = 1001
        is_terminal = True
        reward = 1.0
    elif current_state <= 0:
        current_state = 0
        is_terminal = True
        reward = -1.0
    else:
        is_terminal = False
        reward = 0.0

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    return

def env_message(in_message): 
    return ""
