#!/usr/bin/env python

"""
  Author: Yaozhi Lu
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

gama = 0.95
num_action = 4
board_row = 6
board_col = 9

def agent_init():
    # initialize the policy array in a smart way
    global Q, plan_step, model, old_state, alpha,epsilon

    fp = open("shared.pkl")
    file_info = pickle.load(fp)
    fp.close()

    plan_step = file_info[0]
    alpha = file_info[1]
    epsilon = file_info[2]


    Q = np.zeros((board_col, board_row, num_action))
    model = np.zeros((board_col, board_row, num_action))
    model = model.tolist()

    old_state = []

    return


def agent_start(state):
    # pick the first action, don't forget about exploring starts
    global agent_last_action, agent_last_state
    x = state[0]
    y = state[1]

    if rand_un() < epsilon:
        action = rand_in_range(num_action)
    else:
        if Q[x][y].sum() == 0:
            action = rand_in_range(num_action)
        else:
            action = np.argmax(Q[x][y])

    agent_last_state = [x, y]
    agent_last_action = action

    old_state.append(agent_last_state)

    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    global Q, agent_last_action, agent_last_state, plan_step,old_state,alpha
    x = state[0]
    y = state[1]

    xx = agent_last_state[0]
    yy = agent_last_state[1]

    # select an action, based on Q
    if rand_un() < epsilon:
        action = rand_in_range(num_action)
    else:
        # pick the largest q in Q
        if Q[x][y].sum() == 0:
            action = rand_in_range(num_action)
        else:
            action = np.argmax(Q[x][y])

    Q_SA = Q[xx][yy][agent_last_action]
    Q[xx][yy][agent_last_action] = Q_SA + alpha * (reward + gama * np.max(Q[x][y]) - Q_SA)

    model[xx][yy][agent_last_action] = [reward, x, y]

    for i in range(0, plan_step):
        choice = []
        while choice == []:
            random_state = old_state[rand_in_range(len(old_state))]
            for index in range(0, 4):
                if model[random_state[0]][random_state[1]][index] != 0:
                    choice.append(index)

        random_action = choice[rand_in_range(len(choice))]
        model_info = model[random_state[0]][random_state[1]][random_action]
        plan_reward = model_info[0]
        plan_x_prime = model_info[1]
        plan_y_prime = model_info[2]

        Q_SA_plan = Q[random_state[0]][random_state[1]][random_action]
        Q[random_state[0]][random_state[1]][random_action] = Q_SA_plan + alpha*(plan_reward + gama*(np.max(Q[plan_x_prime][plan_y_prime])) - Q_SA_plan)

    if agent_last_state not in old_state:
        old_state.append(agent_last_state)

    agent_last_action = action
    agent_last_state = [x, y]
    return action

def agent_end(reward):
    # do learning and update pi
    global Q, agent_last_action, agent_last_state

    xx = agent_last_state[0]
    yy = agent_last_state[1]
    Q_SA = Q[xx][yy][agent_last_action]
    Q[xx][yy][agent_last_action] = Q_SA + alpha*(reward - Q_SA)


    #print model[8][4][0]
    #model[xx][yy][agent_last_action] = [reward, 8, 5]
    return


def agent_cleanup():
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global Q
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return
    else:
        return "I don't know what to return!!"

