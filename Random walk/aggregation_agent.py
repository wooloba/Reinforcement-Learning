#!/usr/bin/env python

"""
  Author: Yaozhi Lu

"""

from utils import random_policy,rand_in_range, rand_un
import numpy as np
import pickle


alpha = 0.1
gama = 1.0

def agent_init():
    # input policy here
    global policy,w,xs
    # 10 groups,10 weights
    xs = np.identity(10)
    w = np.zeros(10)

    return



def agent_start(state):
    # pick the first action, don't forget about exploring starts
    global agent_last_state,w,xs

    #take action base on policy
    action = random_policy()
    agent_last_state = state
    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    global w,xs,agent_last_state

    vhat_raw = xs[(state-1)/100]
    last_vhat_raw = xs[(agent_last_state-1)/100]

    deri_vhat = xs[(agent_last_state-1)/100]

    vhat_raw = np.dot(w, vhat_raw)
    vhat_raw *= gama

    last_vhat_raw = np.dot(w, last_vhat_raw)

    w = w + alpha*(reward + vhat_raw - last_vhat_raw) * deri_vhat

    action = random_policy()
    agent_last_state = state

    return action


def agent_end(reward):
    # do learning and update pi
    global w,xs,agent_last_state

    last_vhat_raw = xs[(agent_last_state-1)/100]
    deri_vhat = xs[(agent_last_state-1)/100]

    last_vhat_raw = np.dot(w, last_vhat_raw)

    w = w + alpha * (reward - last_vhat_raw) * deri_vhat

    fp = open("shared.pkl", "w")
    fp.truncate()
    pickle.dump(w, fp)
    fp.close()
    return


def agent_cleanup():
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return
    else:
        return "I don't know what to return!!"