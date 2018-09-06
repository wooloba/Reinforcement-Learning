#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""
from rl_glue import *  # Required for RL-Glue
import matplotlib.pyplot as plt
RLGlue("windy_env", "sarsa_agent")

if __name__ == "__main__":
    max_steps = 8000
    time_steps = []
    episodeData = []
    total_steps = 0
    total_episode = 0
    time_steps.append(total_episode)
    episodeData.append(total_episode)

    stats = []
    episodes = []
    for run in range(0,100):
        RL_init()
        while total_steps < 8000:
            RL_episode(max_steps)
            total_steps += RL_num_steps()
            total_episode = RL_num_episodes()
            time_steps.append(total_steps)
            episodeData.append(total_episode)
            print total_steps, total_episode
        RL_cleanup()


    plt.figure()
    plt.plot(time_steps,episodeData)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()