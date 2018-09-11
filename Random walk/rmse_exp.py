#!/usr/bin/env python

from rl_glue import *  # Required for RL-Glue
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from rndmwalk_policy_evaluation import compute_value_function

if __name__ == "__main__":

    plt.figure()
    episodes = []
    print "If you dont have TrueValueFunction.npy, please comment out line 16 in rmse_exp and uncomment line 15 in rmse_exp"
    # true_value = compute_value_function()
    true_value = np.load("TrueValueFunction.npy")


    episode_num = 5000
    run_num = 10

    def rmse_cal(agg):
        rmse = np.zeros(episode_num)
        for run in range(0, run_num):
            RL_init()
            print "   run: " + str(run + 1),
            np.random.seed(run)
            for episode in range(0, episode_num):
                RL_episode(1500)
                fp = open("shared.pkl")
                w = pickle.load(fp)
                fp.close()

                square_error = 0
                if agg == False:
                    for s in range(0, 1000):
                        square_error += (true_value[s] - w[s]) ** 2
                if agg == True:
                    for s in range(0, 1000):
                        square_error += (true_value[s] - w[s / 100]) ** 2

                rmse_new = math.sqrt(square_error / 1000.0)

                rmse[episode] = (rmse[episode] * run + rmse_new) / float(run + 1)
            RL_cleanup()
            print '.'
        return rmse

    RLGlue("random_walk_env", "tile_coding_agent")
    print ("For tile coding agent: ")
    value = rmse_cal(False)
    n_zero, = plt.plot(range(1, episode_num + 1), value, 'g-', label="tile coding")

    RLGlue("random_walk_env", "tabular_agent")
    print ("For tabular agent: ")
    value = rmse_cal(False)
    n_zero, = plt.plot(range(1, episode_num+1),value,'b-',label="tabular")

    RLGlue("random_walk_env","aggregation_agent")
    print ("For aggregation state agent: ")
    value = rmse_cal(True)
    n_zero, = plt.plot(range(1, episode_num+1), value, 'r-',label = "aggregation")

    print "Done!"
    axes = plt.gca()
    axes.set_xlim([0, episode_num])
    axes.set_ylim([0, 0.6])
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('RMSVE')
    plt.show()