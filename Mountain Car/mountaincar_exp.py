#!/usr/bin/env python

"""
  Author: Yaozhi Lu
"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")
import numpy as np

if __name__ == "__main__":
    num_episodes = 200
    num_runs = 50

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r+1
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            steps[r,e] = RL_num_steps()
    np.save('steps',steps)
    RL_cleanup()
