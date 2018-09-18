#!/usr/bin/env python

"""
  Author: Yaozhi Lu

"""
from rl_glue import *  # Required for RL-Glue
import matplotlib.pyplot as plt
import numpy as np
import pickle
RLGlue("maze_env", "Dyna_Q_agent")

if __name__ == "__main__":

    n_list = [0,5,50]
    alpha = 0.1
    epsilon = 0.1

    plt.figure()
    for item in n_list:
        n = item
        print ("n = " + str(n))

        fp = open("shared.pkl", "w")
        fp.truncate()
        pickle.dump([n,alpha,epsilon], fp)
        fp.close()

        total_steps = 0
        episodes = []
        steps_per_episode = []
        steps_data = np.zeros(50)

        for run in range(0,10):
            np.random.seed(run)
            RL_init()
            print "run: " + str(run+1)
            for episode in range(0,50):
                RL_episode(1500)
                total_steps = RL_num_steps()
                steps_data[episode]+=total_steps
            RL_cleanup()
        for i in range(1,50):
            steps_per_episode.append(steps_data[i]/10.0)
            episodes.append(i+1)
        if n == 0:
            n_zero, = plt.plot(episodes,steps_per_episode,'b-',label="n = 0" )
        elif n == 5:
            n_five, = plt.plot(episodes,steps_per_episode,'g-',label = "n = 5")
        elif n == 50:
            n_fifty, =plt.plot(episodes,steps_per_episode,'r-',label = "n = 50")

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('steps per episode')
    plt.show()